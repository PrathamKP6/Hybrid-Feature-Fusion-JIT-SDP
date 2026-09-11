"""
EXPERIMENT 4B1: REDUCED FEATURE FUSION
======================================

12 JIT features + first 25 PCA-reduced CodeBERT components

Methodology:
- Project-wise chronological 70/15/15 split
- PCA was already fitted on TRAIN ONLY
- Uses first 25 components from existing 384-D PCA representation
- Early feature concatenation
- Join using commit_id
- No resampling
- No random train/test split
- Test set used only for final evaluation

Input:
    results/chronological_splits/train.csv
    results/chronological_splits/validation.csv
    results/chronological_splits/test.csv

    results/pca/train_pca384.csv
    results/pca/validation_pca384.csv
    results/pca/test_pca384.csv

Output:
    results/experiment_4b1_reduced_fusion_25pca/
"""

import ast
import json
import logging
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHRONO_DIR = BASE_DIR / "results" / "chronological_splits"
PCA_DIR = BASE_DIR / "results" / "pca"

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "experiment_4b1_reduced_fusion_25pca"
)

METRICS_DIR = OUTPUT_DIR / "metrics"
MODELS_DIR = OUTPUT_DIR / "models"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
PLOTS_DIR = OUTPUT_DIR / "plots"

for directory in [
    OUTPUT_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    PLOTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)

warnings.filterwarnings("ignore")


# ============================================================
# FEATURES
# ============================================================

JIT_FEATURES = [
    "la",
    "ld",
    "nf",
    "ns",
    "nd",
    "ent",
    "ndev",
    "age",
    "nuc",
    "aexp",
    "arexp",
    "asexp",
]

N_PCA_COMPONENTS = 25
FULL_PCA_DIMENSION = 384

TOTAL_FEATURES = len(JIT_FEATURES) + N_PCA_COMPONENTS


# ============================================================
# FILES
# ============================================================

JIT_FILES = {
    "train": CHRONO_DIR / "train.csv",
    "validation": CHRONO_DIR / "validation.csv",
    "test": CHRONO_DIR / "test.csv",
}

PCA_FILES = {
    "train": PCA_DIR / "train_pca384.csv",
    "validation": PCA_DIR / "validation_pca384.csv",
    "test": PCA_DIR / "test_pca384.csv",
}


# ============================================================
# TARGET NORMALIZATION
# ============================================================

def normalize_target(value):
    """
    Convert different buggy representations to 0/1.
    """

    if pd.isna(value):
        return np.nan

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        return int(value)

    value = str(value).strip().lower()

    if value in {"true", "1", "yes", "y"}:
        return 1

    if value in {"false", "0", "no", "n"}:
        return 0

    try:
        return int(float(value))
    except Exception:
        raise ValueError(
            f"Unable to normalize buggy label: {value}"
        )


# ============================================================
# LOAD JIT DATA
# ============================================================

def load_jit_split(path, split_name):
    logging.info(f"\nLoading JIT {split_name}:")
    logging.info(f"  {path}")

    df = pd.read_csv(path)

    logging.info(f"  Rows: {len(df):,}")
    logging.info(f"  Columns: {len(df.columns)}")

    required = [
        "commit_id",
        "project",
        "buggy",
    ] + JIT_FEATURES

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in JIT {split_name}: {missing}"
        )

    result = df[
        [
            "commit_id",
            "project",
            "buggy",
        ] + JIT_FEATURES
    ].copy()

    result["buggy"] = result["buggy"].apply(normalize_target)

    if result["buggy"].isna().any():
        raise ValueError(
            f"Missing buggy labels found in {split_name}"
        )

    for feature in JIT_FEATURES:
        result[feature] = pd.to_numeric(
            result[feature],
            errors="coerce"
        )

    return result


# ============================================================
# PARSE EMBEDDING
# ============================================================

def parse_embedding(value):
    """
    Parse embedding stored as JSON/list string.
    """

    if isinstance(value, list):
        vector = value

    elif isinstance(value, np.ndarray):
        vector = value.tolist()

    elif pd.isna(value):
        raise ValueError("Embedding is missing")

    else:
        text = str(value).strip()

        try:
            vector = ast.literal_eval(text)
        except Exception:
            try:
                vector = json.loads(text)
            except Exception as exc:
                raise ValueError(
                    f"Unable to parse embedding: {text[:100]}"
                ) from exc

    vector = np.asarray(vector, dtype=np.float32)

    if vector.ndim != 1:
        raise ValueError(
            f"Embedding is not 1-D. Shape: {vector.shape}"
        )

    if len(vector) != FULL_PCA_DIMENSION:
        raise ValueError(
            f"Expected {FULL_PCA_DIMENSION}-D PCA embedding, "
            f"got {len(vector)} dimensions"
        )

    if not np.all(np.isfinite(vector)):
        raise ValueError(
            "Embedding contains NaN or infinite values"
        )

    return vector


# ============================================================
# LOAD PCA DATA
# ============================================================

def load_pca_split(path, split_name):
    logging.info(f"\nLoading CodeBERT PCA {split_name}:")
    logging.info(f"  {path}")

    df = pd.read_csv(path)

    logging.info(f"  Rows: {len(df):,}")
    logging.info(f"  Columns: {len(df.columns)}")

    required = [
        "commit_id",
        "project",
        "buggy",
        "embedding",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in PCA {split_name}: {missing}"
        )

    result = df[required].copy()

    # Normalize buggy label so both JIT and CodeBERT
    # representations use the same 0/1 format.
    result["buggy"] = result["buggy"].apply(
        normalize_target
    )

    if result["buggy"].isna().any():
        raise ValueError(
            f"Missing buggy labels found in PCA {split_name}"
        )

    return result


# ============================================================
# CREATE REDUCED FUSION
# ============================================================

def create_reduced_fused_split(
    jit_df,
    pca_df,
    split_name,
):
    logging.info(
        f"\nCreating {split_name} reduced fused features..."
    )

    # --------------------------------------------------------
    # Verify commit IDs
    # --------------------------------------------------------

    jit_ids = set(jit_df["commit_id"])
    pca_ids = set(pca_df["commit_id"])

    if jit_ids != pca_ids:

        only_jit = jit_ids - pca_ids
        only_pca = pca_ids - jit_ids

        raise ValueError(
            f"Commit ID mismatch in {split_name}:\n"
            f"  Only JIT: {len(only_jit)}\n"
            f"  Only PCA: {len(only_pca)}"
        )

    # --------------------------------------------------------
    # Merge only using commit_id
    # --------------------------------------------------------

    merged = jit_df.merge(
        pca_df,
        on="commit_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_jit", "_pca"),
    )

    # --------------------------------------------------------
    # Verify project and target consistency after commit_id join
    # --------------------------------------------------------

    if not (
        merged["project_jit"].astype(str)
        == merged["project_pca"].astype(str)
    ).all():
        raise ValueError(
            f"Project mismatch detected in {split_name}"
        )

    # --------------------------------------------------------
    # Verify buggy-label consistency
    # --------------------------------------------------------

    if not (
        merged["buggy_jit"].astype(int)
        == merged["buggy_pca"].astype(int)
    ).all():
        raise ValueError(
            f"Buggy label mismatch detected in {split_name}"
        )

    # Use the JIT-side verified target/project
    merged["project"] = merged["project_jit"]
    merged["buggy"] = merged["buggy_jit"]

    # --------------------------------------------------------
    # Verify row count
    # --------------------------------------------------------

    if len(merged) != len(jit_df):
        raise ValueError(
            f"Merge changed row count for {split_name}: "
            f"{len(jit_df)} → {len(merged)}"
        )

    # --------------------------------------------------------
    # JIT matrix
    # --------------------------------------------------------

    jit_matrix = merged[
        JIT_FEATURES
    ].to_numpy(dtype=np.float32)

    # --------------------------------------------------------
    # Parse full 384-D PCA embeddings
    # --------------------------------------------------------

    logging.info(
        f"  Parsing {FULL_PCA_DIMENSION}-D PCA embeddings..."
    )

    full_embeddings = np.vstack(
        merged["embedding"].apply(parse_embedding).values
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Select FIRST 25 components
    #
    # These are components 1–25 from the SAME PCA that was
    # already fitted using training data only.
    # --------------------------------------------------------

    codebert_matrix = full_embeddings[
        :, :N_PCA_COMPONENTS
    ]

    # --------------------------------------------------------
    # Concatenate
    # --------------------------------------------------------

    fused_matrix = np.hstack(
        [
            jit_matrix,
            codebert_matrix,
        ]
    ).astype(np.float32)

    logging.info(
        f"  JIT dimensions: {jit_matrix.shape[1]}"
    )

    logging.info(
        f"  CodeBERT dimensions used: "
        f"{codebert_matrix.shape[1]}"
    )

    logging.info(
        f"  Fused dimensions: {fused_matrix.shape[1]}"
    )

    if fused_matrix.shape[1] != TOTAL_FEATURES:
        raise ValueError(
            f"Expected {TOTAL_FEATURES} fused features, "
            f"got {fused_matrix.shape[1]}"
        )

    return (
        fused_matrix,
        merged["buggy"].to_numpy(dtype=np.int32),
        merged["commit_id"].to_numpy(),
    )


# ============================================================
# IMPUTATION
# ============================================================

def calculate_train_medians(X_train):
    """
    Calculate JIT feature medians from training data only.
    CodeBERT PCA values are expected to be finite.
    """

    medians = np.nanmedian(
        X_train[:, :len(JIT_FEATURES)],
        axis=0
    )

    return medians


def apply_jit_imputation(
    X_train,
    X_val,
    X_test,
):
    """
    Apply training-derived medians to JIT features.
    """

    logging.info("\n" + "=" * 70)
    logging.info("TRAINING-DERIVED IMPUTATION")
    logging.info("=" * 70)

    logging.info(
        "Missing-value handling: "
        "TRAIN-DERIVED JIT MEDIANS ONLY"
    )

    medians = calculate_train_medians(X_train)

    for X in [
        X_train,
        X_val,
        X_test,
    ]:

        for i, median in enumerate(medians):

            mask = ~np.isfinite(X[:, i])

            if np.any(mask):
                X[mask, i] = median

    # Final safety check
    for name, X in [
        ("train", X_train),
        ("validation", X_val),
        ("test", X_test),
    ]:

        if not np.all(np.isfinite(X)):
            raise ValueError(
                f"Non-finite values remain in {name}"
            )

    return medians


# ============================================================
# MODEL CONFIGURATION
# ============================================================

def build_models(y_train):
    negative = np.sum(y_train == 0)
    positive = np.sum(y_train == 1)

    scale_pos_weight = (
        negative / positive
        if positive > 0
        else 1.0
    )

    logging.info("\n" + "=" * 70)
    logging.info("MODEL CONFIGURATION")
    logging.info("=" * 70)

    logging.info(
        f"XGBoost scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    return {
        "random_forest": rf,
        "xgboost": xgb,
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    split_name,
    model_name,
):
    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y,
        predictions
    )

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    pr_auc = average_precision_score(
        y,
        probabilities
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    logging.info("\n" + "=" * 70)
    logging.info(
        f"{split_name.upper()} RESULTS"
    )
    logging.info("=" * 70)

    logging.info(
        f"ACCURACY    : {accuracy:.4f}"
    )

    logging.info(
        f"PRECISION   : {precision:.4f}"
    )

    logging.info(
        f"RECALL      : {recall:.4f}"
    )

    logging.info(
        f"F1          : {f1:.4f}"
    )

    logging.info(
        f"MCC         : {mcc:.4f}"
    )

    logging.info(
        f"ROC_AUC     : {roc_auc:.4f}"
    )

    logging.info(
        f"PR_AUC      : {pr_auc:.4f}"
    )

    logging.info("\nConfusion Matrix:")
    logging.info(cm)

    return {
        "model": model_name,
        "split": split_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": cm[0, 0],
        "fp": cm[0, 1],
        "fn": cm[1, 0],
        "tp": cm[1, 1],
        "probabilities": probabilities,
        "predictions": predictions,
        "confusion_matrix": cm,
    }


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    commit_ids,
    y_true,
    probabilities,
    predictions,
    model_name,
):
    output = pd.DataFrame(
        {
            "commit_id": commit_ids,
            "actual_buggy": y_true,
            "predicted_buggy": predictions,
            "predicted_probability": probabilities,
        }
    )

    path = (
        PREDICTIONS_DIR
        / f"{model_name}_test_predictions.csv"
    )

    output.to_csv(
        path,
        index=False
    )

    logging.info(
        f"Predictions saved: {path}"
    )


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix_plot(
    cm,
    model_name,
):
    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.imshow(cm)

    ax.set_title(
        f"4B1 Reduced Fusion - {model_name}"
    )

    ax.set_xlabel(
        "Predicted Label"
    )

    ax.set_ylabel(
        "Actual Label"
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["Non-Buggy", "Buggy"]
    )

    ax.set_yticklabels(
        ["Non-Buggy", "Buggy"]
    )

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
            )

    plt.tight_layout()

    path = (
        PLOTS_DIR
        / f"{model_name}_confusion_matrix.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# ROC CURVE
# ============================================================

def save_roc_curve(
    y_true,
    probability,
    model_name,
):
    fpr, tpr, _ = roc_curve(
        y_true,
        probability
    )

    auc = roc_auc_score(
        y_true,
        probability
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        f"4B1 Reduced Fusion ROC Curve - {model_name}"
    )

    plt.legend()

    plt.tight_layout()

    path = (
        PLOTS_DIR
        / f"{model_name}_roc_curve.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# PR CURVE
# ============================================================

def save_pr_curve(
    y_true,
    probability,
    model_name,
):
    precision, recall, _ = precision_recall_curve(
        y_true,
        probability
    )

    auc = average_precision_score(
        y_true,
        probability
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        recall,
        precision,
        label=f"AP = {auc:.4f}"
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        f"4B1 Reduced Fusion PR Curve - {model_name}"
    )

    plt.legend()

    plt.tight_layout()

    path = (
        PLOTS_DIR
        / f"{model_name}_pr_curve.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    logging.info("\n")
    logging.info("=" * 70)
    logging.info(
        "EXPERIMENT 4B1: REDUCED FEATURE FUSION"
    )
    logging.info("=" * 70)

    logging.info("\nMethodology:")
    logging.info(
        "  Split: Project-wise chronological 70/15/15"
    )
    logging.info(
        "  JIT features: 12"
    )
    logging.info(
        "  CodeBERT features: FIRST 25 of existing 384 PCA components"
    )
    logging.info(
        "  Total fused features: 37"
    )
    logging.info(
        "  PCA: Existing PCA fitted on TRAIN ONLY"
    )
    logging.info(
        "  Fusion: Early feature concatenation"
    )
    logging.info(
        "  Join key: commit_id"
    )
    logging.info(
        "  Resampling: NOT USED"
    )
    logging.info(
        "  Random train/test split: NOT USED"
    )
    logging.info(
        "  Test set: Used only for final evaluation"
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    jit = {}
    pca = {}

    for split in [
        "train",
        "validation",
        "test",
    ]:
        jit[split] = load_jit_split(
            JIT_FILES[split],
            split
        )

        pca[split] = load_pca_split(
            PCA_FILES[split],
            split
        )

    # ========================================================
    # SPLIT VERIFICATION
    # ========================================================

    logging.info("\n" + "=" * 70)
    logging.info("SPLIT VERIFICATION")
    logging.info("=" * 70)

    logging.info(
        f"Train:       {len(jit['train']):,}"
    )

    logging.info(
        f"Validation:  {len(jit['validation']):,}"
    )

    logging.info(
        f"Test:        {len(jit['test']):,}"
    )

    total = sum(
        len(jit[s])
        for s in [
            "train",
            "validation",
            "test",
        ]
    )

    logging.info(
        f"Total:       {total:,}"
    )

    train_ids = set(
        jit["train"]["commit_id"]
    )

    val_ids = set(
        jit["validation"]["commit_id"]
    )

    test_ids = set(
        jit["test"]["commit_id"]
    )

    overlap = (
        (train_ids & val_ids)
        | (train_ids & test_ids)
        | (val_ids & test_ids)
    )

    if overlap:
        raise ValueError(
            f"Commit IDs overlap across splits: "
            f"{len(overlap)}"
        )

    logging.info(
        "Commit ID overlap: NONE"
    )

    # ========================================================
    # TARGET DISTRIBUTION
    # ========================================================

    logging.info("\n" + "=" * 70)
    logging.info("TARGET DISTRIBUTION")
    logging.info("=" * 70)

    for split in [
        "train",
        "validation",
        "test",
    ]:
        y = jit[split]["buggy"]

        positive = int(
            np.sum(y == 1)
        )

        logging.info(
            f"{split.capitalize():12s}: "
            f"buggy={positive:,} / {len(y):,} "
            f"({positive / len(y) * 100:.2f}%)"
        )

    # ========================================================
    # CREATE REDUCED FUSION
    # ========================================================

    fused = {}
    targets = {}
    commit_ids = {}

    for split in [
        "train",
        "validation",
        "test",
    ]:

        (
            fused[split],
            targets[split],
            commit_ids[split],
        ) = create_reduced_fused_split(
            jit[split],
            pca[split],
            split
        )

    # ========================================================
    # FEATURE VERIFICATION
    # ========================================================

    logging.info("\n" + "=" * 70)
    logging.info("FEATURE DIMENSION VERIFICATION")
    logging.info("=" * 70)

    logging.info(
        f"JIT features:       {len(JIT_FEATURES)}"
    )

    logging.info(
        f"CodeBERT features:  {N_PCA_COMPONENTS}"
    )

    logging.info(
        f"Total features:     {TOTAL_FEATURES}"
    )

    logging.info(
        f"Train matrix:       {fused['train'].shape}"
    )

    logging.info(
        f"Validation matrix:  {fused['validation'].shape}"
    )

    logging.info(
        f"Test matrix:        {fused['test'].shape}"
    )

    assert fused["train"].shape[1] == 37
    assert fused["validation"].shape[1] == 37
    assert fused["test"].shape[1] == 37

    # ========================================================
    # IMPUTATION
    # ========================================================

    medians = apply_jit_imputation(
        fused["train"],
        fused["validation"],
        fused["test"],
    )

    # ========================================================
    # SAVE CONFIG
    # ========================================================

    config = {
        "experiment": "4B1",
        "description": "Reduced Feature Fusion",
        "jit_features": JIT_FEATURES,
        "jit_dimension": len(JIT_FEATURES),
        "full_pca_dimension": FULL_PCA_DIMENSION,
        "selected_pca_components": N_PCA_COMPONENTS,
        "total_fused_features": TOTAL_FEATURES,
        "fusion_strategy": "early_concatenation",
        "pca_fitted_on": "train_only",
        "pca_selection": "first_25_components_of_existing_384d_pca",
        "join_key": "commit_id",
        "resampling": False,
        "random_split": False,
        "threshold": 0.5,
        "random_state": 42,
        "rf_estimators": 300,
        "xgb_estimators": 300,
    }

    with open(
        METRICS_DIR / "config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2
        )

    # ========================================================
    # TRAIN MODELS
    # ========================================================

    models = build_models(
        targets["train"]
    )

    all_results = []

    for model_name, model in models.items():

        logging.info("\n")
        logging.info("#" * 70)
        logging.info(
            f"TRAINING: {model_name.upper()}"
        )
        logging.info("#" * 70)

        model.fit(
            fused["train"],
            targets["train"]
        )

        logging.info(
            "Training completed."
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_result = evaluate_model(
            model,
            fused["validation"],
            targets["validation"],
            "validation",
            model_name,
        )

        all_results.append({
            "model": model_name,
            "split": "validation",
            "accuracy": val_result["accuracy"],
            "precision": val_result["precision"],
            "recall": val_result["recall"],
            "f1": val_result["f1"],
            "mcc": val_result["mcc"],
            "roc_auc": val_result["roc_auc"],
            "pr_auc": val_result["pr_auc"],
            "tn": val_result["tn"],
            "fp": val_result["fp"],
            "fn": val_result["fn"],
            "tp": val_result["tp"],
        })

        # ----------------------------------------------------
        # Test
        # ----------------------------------------------------

        test_result = evaluate_model(
            model,
            fused["test"],
            targets["test"],
            "test",
            model_name,
        )

        all_results.append({
            "model": model_name,
            "split": "test",
            "accuracy": test_result["accuracy"],
            "precision": test_result["precision"],
            "recall": test_result["recall"],
            "f1": test_result["f1"],
            "mcc": test_result["mcc"],
            "roc_auc": test_result["roc_auc"],
            "pr_auc": test_result["pr_auc"],
            "tn": test_result["tn"],
            "fp": test_result["fp"],
            "fn": test_result["fn"],
            "tp": test_result["tp"],
        })

        # ----------------------------------------------------
        # Save model
        # ----------------------------------------------------

        model_path = (
            MODELS_DIR
            / f"{model_name}_jit_codebert_25pca.joblib"
        )

        joblib.dump(
            model,
            model_path
        )

        logging.info(
            f"\nModel saved: {model_path}"
        )

        # ----------------------------------------------------
        # Save test predictions
        # ----------------------------------------------------

        save_predictions(
            commit_ids["test"],
            targets["test"],
            test_result["probabilities"],
            test_result["predictions"],
            model_name,
        )

        # ----------------------------------------------------
        # Save plots
        # ----------------------------------------------------

        save_confusion_matrix_plot(
            test_result["confusion_matrix"],
            model_name,
        )

        save_roc_curve(
            targets["test"],
            test_result["probabilities"],
            model_name,
        )

        save_pr_curve(
            targets["test"],
            test_result["probabilities"],
            model_name,
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    results_path = (
        METRICS_DIR
        / "experiment_4b1_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    # ========================================================
    # FINAL TEST SUMMARY
    # ========================================================

    test_results = results_df[
        results_df["split"] == "test"
    ].copy()

    logging.info("\n" + "=" * 70)
    logging.info(
        "EXPERIMENT 4B1 COMPLETED"
    )
    logging.info("=" * 70)

    logging.info("\nFinal TEST performance:")

    display_columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "roc_auc",
        "pr_auc",
    ]

    logging.info(
        "\n" +
        test_results[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # COMPARISON WITH 4A
    # ========================================================

    logging.info("\n" + "=" * 70)
    logging.info(
        "COMPARISON WITH EXPERIMENT 4A"
    )
    logging.info("=" * 70)

    # Existing Experiment 4A test results
    exp4a = pd.DataFrame([
        {
            "model": "Random Forest",
            "f1": 0.424400,
            "mcc": 0.304189,
            "roc_auc": 0.777167,
            "pr_auc": 0.367648,
        },
        {
            "model": "XGBoost",
            "f1": 0.449094,
            "mcc": 0.348532,
            "roc_auc": 0.803730,
            "pr_auc": 0.403761,
        },
    ])

    comparison_rows = []

    for _, row in test_results.iterrows():

        model_label = (
            "Random Forest"
            if row["model"] == "random_forest"
            else "XGBoost"
        )

        baseline = exp4a[
            exp4a["model"] == model_label
        ].iloc[0]

        comparison_rows.append({
            "model": model_label,
            "4A_F1": baseline["f1"],
            "4B1_F1": row["f1"],
            "delta_F1": row["f1"] - baseline["f1"],
            "4A_MCC": baseline["mcc"],
            "4B1_MCC": row["mcc"],
            "delta_MCC": row["mcc"] - baseline["mcc"],
            "4A_ROC_AUC": baseline["roc_auc"],
            "4B1_ROC_AUC": row["roc_auc"],
            "delta_ROC_AUC": row["roc_auc"] - baseline["roc_auc"],
            "4A_PR_AUC": baseline["pr_auc"],
            "4B1_PR_AUC": row["pr_auc"],
            "delta_PR_AUC": row["pr_auc"] - baseline["pr_auc"],
        })

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_path = (
        METRICS_DIR
        / "experiment_4a_vs_4b1_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    logging.info(
        "\n" +
        comparison_df.to_string(
            index=False
        )
    )

    # ========================================================
    # FINAL FILE LOCATIONS
    # ========================================================

    logging.info("\n" + "=" * 70)
    logging.info("OUTPUT FILES")
    logging.info("=" * 70)

    logging.info(
        f"Results:      {results_path}"
    )

    logging.info(
        f"Comparison:   {comparison_path}"
    )

    logging.info(
        f"Models:       {MODELS_DIR}"
    )

    logging.info(
        f"Predictions:  {PREDICTIONS_DIR}"
    )

    logging.info(
        f"Plots:        {PLOTS_DIR}"
    )

    logging.info(
        f"Config:       {METRICS_DIR / 'config.json'}"
    )

    logging.info("\n")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()