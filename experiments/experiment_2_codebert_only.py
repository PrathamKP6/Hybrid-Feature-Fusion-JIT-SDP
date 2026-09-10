"""
Experiment 2: CodeBERT-Only Baseline
====================================

Uses the leakage-safe 384-D PCA embeddings generated previously.

Input:
    results/pca/train_pca384.csv
    results/pca/validation_pca384.csv
    results/pca/test_pca384.csv

The PCA files contain the 384-D PCA representation inside the
existing `embedding` column rather than pca_1 ... pca_384 columns.

Split:
    Project-wise chronological 70/15/15

Features:
    384-D PCA-reduced CodeBERT embeddings ONLY

Models:
    Random Forest
    XGBoost

No JIT/process features are used.
No new PCA is fitted.
No resampling is performed.
"""

from __future__ import annotations

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
    average_precision_score,
    auc,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PCA_DIR = PROJECT_ROOT / "results" / "pca"

TRAIN_PATH = PCA_DIR / "train_pca384.csv"
VALIDATION_PATH = PCA_DIR / "validation_pca384.csv"
TEST_PATH = PCA_DIR / "test_pca384.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "experiment_2_codebert_only"
)

METRICS_DIR = OUTPUT_DIR / "metrics"
MODELS_DIR = OUTPUT_DIR / "models"
PRED_DIR = OUTPUT_DIR / "predictions"
PLOTS_DIR = OUTPUT_DIR / "plots"

for directory in [
    METRICS_DIR,
    MODELS_DIR,
    PRED_DIR,
    PLOTS_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# CONSTANTS
# ============================================================

EMBEDDING_COLUMN = "embedding"

EMBEDDING_DIMENSION = 384

TARGET = "buggy"

BOOKKEEPING_COLUMNS = [
    "commit_id",
    "project",
    "author_date",
]


# ============================================================
# TARGET NORMALIZATION
# ============================================================

def normalize_buggy(value):
    """
    Convert all expected buggy-label representations into 0/1.
    """

    if pd.isna(value):
        return np.nan

    if isinstance(value, (bool, np.bool_)):
        return int(value)

    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        if value in (0, 1):
            return int(value)

    value_str = str(value).strip().lower()

    if value_str in {"0", "false"}:
        return 0

    if value_str in {"1", "true"}:
        return 1

    raise ValueError(
        f"Unexpected buggy value: {value!r}"
    )


# ============================================================
# EMBEDDING PARSER
# ============================================================

def parse_embedding(value):
    """
    Parse one 384-D PCA embedding.

    The PCA output is stored in the existing `embedding`
    column as a serialized list.
    """

    if pd.isna(value):
        raise ValueError(
            "Encountered missing embedding."
        )

    # Already a Python list/array
    if isinstance(
        value,
        (list, tuple, np.ndarray),
    ):
        embedding = list(value)

    # Serialized embedding
    elif isinstance(value, str):

        value = value.strip()

        try:
            embedding = ast.literal_eval(value)

        except Exception:

            try:
                embedding = json.loads(value)

            except Exception as exc:

                raise ValueError(
                    "Could not parse embedding."
                ) from exc

    else:

        raise ValueError(
            f"Unsupported embedding type: "
            f"{type(value)}"
        )

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    )

    if embedding.ndim != 1:

        raise ValueError(
            f"Expected 1-D embedding, "
            f"got shape {embedding.shape}"
        )

    if len(embedding) != EMBEDDING_DIMENSION:

        raise ValueError(
            f"Expected {EMBEDDING_DIMENSION}-D embedding, "
            f"got {len(embedding)} dimensions."
        )

    if not np.isfinite(embedding).all():

        raise ValueError(
            "Embedding contains NaN or infinite values."
        )

    return embedding


# ============================================================
# LOAD PCA SPLIT
# ============================================================

def load_pca_split(
    path: Path,
    split_name: str,
):
    """
    Load one PCA-transformed split.

    Returns:
        df
        X = 384-D PCA matrix
        y = binary target
    """

    print(f"\nLoading {split_name}:")
    print(f"  {path}")

    if not path.exists():

        raise FileNotFoundError(
            f"Missing PCA split: {path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    print(
        f"  Rows: {len(df):,}"
    )

    print(
        f"  Columns: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required = (
        BOOKKEEPING_COLUMNS
        + [
            TARGET,
            EMBEDDING_COLUMN,
        ]
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{split_name} is missing required "
            f"columns: {missing}"
        )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    df[TARGET] = df[TARGET].apply(
        normalize_buggy
    )

    if df[TARGET].isna().any():

        count = int(
            df[TARGET].isna().sum()
        )

        raise ValueError(
            f"{split_name} contains "
            f"{count} missing buggy labels."
        )

    df[TARGET] = df[TARGET].astype(int)

    # --------------------------------------------------------
    # Duplicate IDs
    # --------------------------------------------------------

    duplicate_ids = int(
        df["commit_id"].duplicated().sum()
    )

    if duplicate_ids > 0:

        raise ValueError(
            f"{split_name} contains "
            f"{duplicate_ids} duplicate commit IDs."
        )

    # --------------------------------------------------------
    # Parse embeddings
    # --------------------------------------------------------

    print(
        f"  Parsing {EMBEDDING_DIMENSION}-D "
        f"PCA embeddings..."
    )

    embeddings = []

    for index, value in enumerate(
        df[EMBEDDING_COLUMN]
    ):

        try:

            embedding = parse_embedding(
                value
            )

            embeddings.append(
                embedding
            )

        except Exception as exc:

            raise ValueError(
                f"Invalid embedding in "
                f"{split_name} at row {index}: "
                f"{exc}"
            ) from exc

    X = np.vstack(
        embeddings
    ).astype(np.float32)

    y = df[TARGET].to_numpy(
        dtype=np.int32
    )

    print(
        f"  Feature matrix: "
        f"{X.shape[0]:,} × {X.shape[1]}"
    )

    if X.shape[1] != EMBEDDING_DIMENSION:

        raise ValueError(
            f"Expected {EMBEDDING_DIMENSION} "
            f"features, got {X.shape[1]}"
        )

    return df, X, y


# ============================================================
# VERIFY SPLITS
# ============================================================

def verify_splits(
    train_df,
    validation_df,
    test_df,
):
    print("\n" + "=" * 70)
    print("SPLIT VERIFICATION")
    print("=" * 70)

    print(
        f"Train:       {len(train_df):,}"
    )

    print(
        f"Validation:  {len(validation_df):,}"
    )

    print(
        f"Test:        {len(test_df):,}"
    )

    total = (
        len(train_df)
        + len(validation_df)
        + len(test_df)
    )

    print(
        f"Total:       {total:,}"
    )

    if total != 59996:

        raise ValueError(
            f"Expected 59,996 rows, "
            f"found {total:,}."
        )

    train_ids = set(
        train_df["commit_id"]
    )

    validation_ids = set(
        validation_df["commit_id"]
    )

    test_ids = set(
        test_df["commit_id"]
    )

    if train_ids & validation_ids:

        raise ValueError(
            "Train/validation commit overlap detected."
        )

    if train_ids & test_ids:

        raise ValueError(
            "Train/test commit overlap detected."
        )

    if validation_ids & test_ids:

        raise ValueError(
            "Validation/test commit overlap detected."
        )

    print(
        "Commit ID overlap: NONE"
    )


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

def print_target_distribution(
    train_df,
    validation_df,
    test_df,
):
    print("\n" + "=" * 70)
    print("TARGET DISTRIBUTION")
    print("=" * 70)

    for name, df in [
        ("Train", train_df),
        ("Validation", validation_df),
        ("Test", test_df),
    ]:

        buggy_count = int(
            df[TARGET].sum()
        )

        total = len(df)

        percentage = (
            buggy_count / total * 100
        )

        print(
            f"{name:12s}: "
            f"buggy={buggy_count:,} / "
            f"{total:,} "
            f"({percentage:.2f}%)"
        )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_probability,
):

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "mcc": float(
            matthews_corrcoef(
                y_true,
                y_pred,
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_true,
                y_probability,
            )
        ),

        "pr_auc": float(
            average_precision_score(
                y_true,
                y_probability,
            )
        ),
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    split_name,
):

    probability = model.predict_proba(
        X
    )[:, 1]

    prediction = (
        probability >= 0.5
    ).astype(int)

    metrics = calculate_metrics(
        y,
        prediction,
        probability,
    )

    cm = confusion_matrix(
        y,
        prediction,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        f"{split_name.upper()} RESULTS"
    )

    print(
        "=" * 70
    )

    for metric, value in metrics.items():

        print(
            f"{metric.upper():12s}: "
            f"{value:.4f}"
        )

    print("\nConfusion Matrix:")

    print(cm)

    return (
        metrics,
        prediction,
        probability,
        cm,
    )


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    path: Path,
    data,
):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
        )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    df,
    predictions,
    probabilities,
    model_name,
    split_name,
):

    output = df[
        [
            "commit_id",
            "project",
            "author_date",
            TARGET,
        ]
    ].copy()

    output["prediction"] = predictions

    output["probability"] = probabilities

    output_path = (
        PRED_DIR
        / f"{model_name}_{split_name}_predictions.csv"
    )

    output.to_csv(
        output_path,
        index=False,
    )

    return output_path


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    cm,
    model_name,
    split_name,
):

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual_0",
            "Actual_1",
        ],
        columns=[
            "Predicted_0",
            "Predicted_1",
        ],
    )

    path = (
        METRICS_DIR
        / f"{model_name}_{split_name}_confusion_matrix.csv"
    )

    cm_df.to_csv(path)

    return path


# ============================================================
# ROC + PR PLOTS
# ============================================================

def plot_roc_pr(
    y_true,
    probability,
    model_name,
    split_name,
):

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    fpr, tpr, _ = roc_curve(
        y_true,
        probability,
    )

    roc_auc = auc(
        fpr,
        tpr,
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.4f}",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        f"ROC Curve - "
        f"{model_name} - "
        f"{split_name}"
    )

    plt.legend(
        loc="lower right"
    )

    plt.tight_layout()

    roc_path = (
        PLOTS_DIR
        / f"{model_name}_{split_name}_roc.png"
    )

    plt.savefig(
        roc_path,
        dpi=150,
    )

    plt.close()

    # --------------------------------------------------------
    # Precision-Recall
    # --------------------------------------------------------

    precision, recall, _ = (
        precision_recall_curve(
            y_true,
            probability,
        )
    )

    pr_auc = auc(
        recall,
        precision,
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        recall,
        precision,
        label=f"AUC = {pr_auc:.4f}",
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        f"Precision-Recall Curve - "
        f"{model_name} - "
        f"{split_name}"
    )

    plt.legend(
        loc="lower left"
    )

    plt.tight_layout()

    pr_path = (
        PLOTS_DIR
        / f"{model_name}_{split_name}_pr.png"
    )

    plt.savefig(
        pr_path,
        dpi=150,
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: CODEBERT-ONLY BASELINE")
    print("=" * 70)

    print("\nMethodology:")

    print(
        "  Split: Project-wise chronological 70/15/15"
    )

    print(
        "  Features: 384-D PCA-reduced CodeBERT"
    )

    print(
        "  PCA: Fitted on TRAIN ONLY"
    )

    print(
        "  Embedding dimension: 384"
    )

    print(
        "  JIT features: NOT USED"
    )

    print(
        "  Resampling: NOT USED"
    )

    print(
        "  Random train/test split: NOT USED"
    )

    print(
        "  Test set: Used only for final evaluation"
    )

    print("=" * 70)


    # ========================================================
    # 1. LOAD SPLITS
    # ========================================================

    train_df, X_train, y_train = (
        load_pca_split(
            TRAIN_PATH,
            "train",
        )
    )

    validation_df, X_validation, y_validation = (
        load_pca_split(
            VALIDATION_PATH,
            "validation",
        )
    )

    test_df, X_test, y_test = (
        load_pca_split(
            TEST_PATH,
            "test",
        )
    )


    # ========================================================
    # 2. VERIFY SPLITS
    # ========================================================

    verify_splits(
        train_df,
        validation_df,
        test_df,
    )


    # ========================================================
    # 3. TARGET DISTRIBUTION
    # ========================================================

    print_target_distribution(
        train_df,
        validation_df,
        test_df,
    )


    # ========================================================
    # 4. CLASS IMBALANCE
    # ========================================================

    negative_count = np.sum(
        y_train == 0
    )

    positive_count = np.sum(
        y_train == 1
    )

    if positive_count == 0:

        raise ValueError(
            "Training set contains no positive examples."
        )

    scale_pos_weight = (
        negative_count
        / positive_count
    )

    print("\n" + "=" * 70)
    print("MODEL CONFIGURATION")
    print("=" * 70)

    print(
        f"XGBoost scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )


    # ========================================================
    # 5. DEFINE MODELS
    # ========================================================

    models = {

        "random_forest":
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),

        "xgboost":
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
            ),
    }


    # ========================================================
    # 6. TRAIN + EVALUATE
    # ========================================================

    all_results = []


    for model_name, model in models.items():

        print(
            "\n\n" + "#" * 70
        )

        print(
            f"TRAINING: "
            f"{model_name.upper()}"
        )

        print(
            "#" * 70
        )


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train,
        )

        print(
            "Training completed."
        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        (
            validation_metrics,
            validation_predictions,
            validation_probabilities,
            validation_cm,
        ) = evaluate_model(
            model,
            X_validation,
            y_validation,
            "validation",
        )

        save_predictions(
            validation_df,
            validation_predictions,
            validation_probabilities,
            model_name,
            "validation",
        )

        save_confusion_matrix(
            validation_cm,
            model_name,
            "validation",
        )

        plot_roc_pr(
            y_validation,
            validation_probabilities,
            model_name,
            "validation",
        )


        # ----------------------------------------------------
        # TEST
        # ----------------------------------------------------

        (
            test_metrics,
            test_predictions,
            test_probabilities,
            test_cm,
        ) = evaluate_model(
            model,
            X_test,
            y_test,
            "test",
        )

        save_predictions(
            test_df,
            test_predictions,
            test_probabilities,
            model_name,
            "test",
        )

        save_confusion_matrix(
            test_cm,
            model_name,
            "test",
        )

        plot_roc_pr(
            y_test,
            test_probabilities,
            model_name,
            "test",
        )


        # ----------------------------------------------------
        # SAVE MODEL
        # ----------------------------------------------------

        model_path = (
            MODELS_DIR
            / f"{model_name}_codebert.joblib"
        )

        joblib.dump(
            model,
            model_path,
        )

        print(
            f"\nModel saved: "
            f"{model_path}"
        )


        # ----------------------------------------------------
        # STORE RESULTS
        # ----------------------------------------------------

        all_results.append(
            {
                "model": model_name,

                "validation_accuracy":
                    validation_metrics["accuracy"],

                "validation_precision":
                    validation_metrics["precision"],

                "validation_recall":
                    validation_metrics["recall"],

                "validation_f1":
                    validation_metrics["f1"],

                "validation_mcc":
                    validation_metrics["mcc"],

                "validation_roc_auc":
                    validation_metrics["roc_auc"],

                "validation_pr_auc":
                    validation_metrics["pr_auc"],

                "test_accuracy":
                    test_metrics["accuracy"],

                "test_precision":
                    test_metrics["precision"],

                "test_recall":
                    test_metrics["recall"],

                "test_f1":
                    test_metrics["f1"],

                "test_mcc":
                    test_metrics["mcc"],

                "test_roc_auc":
                    test_metrics["roc_auc"],

                "test_pr_auc":
                    test_metrics["pr_auc"],
            }
        )


    # ========================================================
    # 7. SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    results_path = (
        METRICS_DIR
        / "experiment_2_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )


    # ========================================================
    # 8. SAVE JSON SUMMARY
    # ========================================================

    summary = {
        row["model"]: {
            key: value
            for key, value in row.items()
            if key != "model"
        }
        for row in all_results
    }

    save_json(
        METRICS_DIR / "summary.json",
        summary,
    )


    # ========================================================
    # 9. SAVE CONFIGURATION
    # ========================================================

    configuration = {

        "experiment":
            "Experiment 2 - CodeBERT Only",

        "split_method":
            "Project-wise chronological 70/15/15",

        "train_rows":
            int(len(train_df)),

        "validation_rows":
            int(len(validation_df)),

        "test_rows":
            int(len(test_df)),

        "feature_type":
            "PCA-reduced CodeBERT embedding",

        "original_embedding_dimension":
            768,

        "pca_dimension":
            384,

        "embedding_storage":
            "embedding column",

        "pca_fitted_on":
            "training split only",

        "jit_features_used":
            False,

        "resampling_used":
            False,

        "random_split_used":
            False,

        "threshold":
            0.5,

        "random_state":
            42,

        "xgboost_scale_pos_weight":
            float(scale_pos_weight),

        "models": [
            "Random Forest",
            "XGBoost",
        ],
    }

    save_json(
        OUTPUT_DIR
        / "experiment_2_configuration.json",
        configuration,
    )


    # ========================================================
    # 10. FINAL SUMMARY
    # ========================================================

    print(
        "\n\n" + "=" * 70
    )

    print(
        "EXPERIMENT 2 COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        "\nFinal TEST performance:"
    )

    display_columns = [
        "model",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_mcc",
        "test_roc_auc",
        "test_pr_auc",
    ]

    print(
        results_df[
            display_columns
        ].to_string(index=False)
    )

    print(
        "\nResults saved to:"
    )

    print(
        f"  {results_path}"
    )

    print(
        "\nOutput directory:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print(
        "\n" + "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()