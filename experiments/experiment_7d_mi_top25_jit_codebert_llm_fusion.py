"""
EXPERIMENT 7D: JIT + CODEBERT + LLM REASONING FEATURE FUSION
                  WITH TRAIN-ONLY MUTUAL INFORMATION TOP-25 SELECTION

Purpose:
    Evaluate whether a compact subset of the complete 447-feature
    three-way fusion can retain predictive information.

Feature configuration:
    Full feature space:
        JIT:       12
        CodeBERT:  384 PCA features
        LLM:       51
        Total:     447

    Selected feature space:
        MI top-25 features, selected using TRAINING DATA ONLY.

Models:
    - Random Forest
    - XGBoost
    - LightGBM

Methodology:
    - Project-wise chronological 70/15/15 split
    - PCA representation already fitted on TRAIN ONLY
    - One-hot encoder fitted on TRAIN ONLY
    - Numerical median imputation fitted on TRAIN ONLY
    - Mutual information fitted on TRAIN ONLY
    - No resampling
    - No random train/test split
    - commit_id used ONLY for alignment/verification
    - Validation used only for reporting
    - Test set used only for final evaluation
    - Classification threshold = 0.5
"""

from pathlib import Path
import ast
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier
import matplotlib.pyplot as plt

try:
    from lightgbm import LGBMClassifier
except ImportError:
    raise ImportError(
        "\nLightGBM is not installed in the current virtual environment.\n"
        "Install it with:\n"
        "    pip install lightgbm\n"
    )

warnings.filterwarnings("ignore")


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CODEBERT_DIR = PROJECT_ROOT / "results" / "pca"
LLM_DIR = PROJECT_ROOT / "data" / "llm_experiment_dataset"
JIT_DIR = PROJECT_ROOT / "results" / "chronological_splits"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "experiment_7d_mi_top25_jit_codebert_llm_fusion"
)

MODEL_DIR = OUTPUT_DIR / "models"
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
PLOTS_DIR = OUTPUT_DIR / "plots"

for directory in [MODEL_DIR, METRICS_DIR, PREDICTIONS_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# INPUT FILES
# ============================================================================

CODEBERT_FILES = {
    "train": CODEBERT_DIR / "train_pca384.csv",
    "validation": CODEBERT_DIR / "validation_pca384.csv",
    "test": CODEBERT_DIR / "test_pca384.csv",
}

LLM_FILES = {
    "train": LLM_DIR / "train.csv",
    "validation": LLM_DIR / "validation.csv",
    "test": LLM_DIR / "test.csv",
}

JIT_FILES = {
    "train": JIT_DIR / "train.csv",
    "validation": JIT_DIR / "validation.csv",
    "test": JIT_DIR / "test.csv",
}


# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

JIT_FEATURES = [
    "la", "ld", "nf", "ns", "nd", "ent",
    "ndev", "age", "nuc", "aexp", "arexp", "asexp",
]

LLM_CATEGORICAL_COLUMNS = [
    "intent",
    "change",
    "risk",
    "complexity",
    "scope",
    "test",
    "security",
]

LLM_NUMERICAL_COLUMNS = [
    "intent_confidence",
    "change_confidence",
    "risk_confidence",
    "complexity_confidence",
    "scope_confidence",
    "test_confidence",
    "security_confidence",
    "intent_margin",
    "change_margin",
    "risk_margin",
    "complexity_margin",
    "scope_margin",
    "test_margin",
    "security_margin",
]

TARGET_COLUMN = "buggy"
ID_COLUMN = "commit_id"

CODEBERT_SOURCE_DIM = 384
LLM_DIM = 51
JIT_DIM = 12
FULL_FEATURES = 447
TOP_K = 25

THRESHOLD = 0.5
RANDOM_STATE = 42


# ============================================================================
# HELPERS
# ============================================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_file_exists(path):
    if not path.exists():
        raise FileNotFoundError(f"\nRequired file does not exist:\n{path}")


def normalize_target(series):
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
        if values.isna().any():
            raise ValueError("Target column contains invalid numeric values.")
        unique_values = set(values.unique())
        if not unique_values.issubset({0, 1}):
            raise ValueError(
                f"Target must contain only 0/1. Found: {sorted(unique_values)}"
            )
        return values.astype(int)

    mapping = {
        "0": 0,
        "1": 1,
        "false": 0,
        "true": 1,
        "non-buggy": 0,
        "nonbuggy": 0,
        "buggy": 1,
        "bug": 1,
        "clean": 0,
    }

    normalized = (
        series.astype(str).str.strip().str.lower().map(mapping)
    )

    if normalized.isna().any():
        bad_values = series[normalized.isna()].unique()
        raise ValueError(f"Unknown target values: {bad_values}")

    return normalized.astype(int)


def verify_ids(name, df):
    if ID_COLUMN not in df.columns:
        raise ValueError(f"{name}: '{ID_COLUMN}' column is missing.")

    duplicate_count = df[ID_COLUMN].duplicated().sum()

    print(
        f"{name}: rows={len(df):,}, "
        f"duplicate commit IDs={duplicate_count:,}"
    )

    if duplicate_count != 0:
        raise ValueError(f"{name} contains duplicate commit IDs.")


def verify_no_overlap(train_df, val_df, test_df):
    train_ids = set(train_df[ID_COLUMN])
    val_ids = set(val_df[ID_COLUMN])
    test_ids = set(test_df[ID_COLUMN])

    train_val = len(train_ids & val_ids)
    train_test = len(train_ids & test_ids)
    val_test = len(val_ids & test_ids)

    print(f"Train / Validation: {train_val}")
    print(f"Train / Test:       {train_test}")
    print(f"Validation / Test:  {val_test}")

    if train_val or train_test or val_test:
        raise ValueError("Cross-split commit overlap detected.")

    print("\n[OK] No cross-split commit overlap.")


def parse_embedding_column(series, expected_dim=384):
    vectors = []

    for idx, value in enumerate(series):
        if isinstance(value, np.ndarray):
            vector = value.tolist()
        elif isinstance(value, list):
            vector = value
        elif isinstance(value, str):
            text = value.strip()
            try:
                vector = json.loads(text)
            except Exception:
                try:
                    vector = ast.literal_eval(text)
                except Exception as exc:
                    raise ValueError(
                        f"Could not parse embedding at row {idx}: {exc}"
                    )
        else:
            raise ValueError(
                f"Unsupported embedding type at row {idx}: {type(value)}"
            )

        if not isinstance(vector, (list, tuple)):
            raise ValueError(f"Embedding at row {idx} is not a list.")

        if len(vector) != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch at row {idx}: "
                f"expected {expected_dim}, got {len(vector)}"
            )

        vectors.append(vector)

    matrix = np.asarray(vectors, dtype=np.float32)

    if matrix.shape[1] != expected_dim:
        raise ValueError(
            f"Embedding matrix has shape {matrix.shape}; "
            f"expected second dimension {expected_dim}."
        )

    return matrix


def align_by_commit_id(reference_df, other_df, name):
    verify_ids(f"Reference {name}", reference_df)
    verify_ids(f"Other {name}", other_df)

    reference_ids = set(reference_df[ID_COLUMN])
    other_ids = set(other_df[ID_COLUMN])

    missing_in_other = reference_ids - other_ids
    missing_in_reference = other_ids - reference_ids

    if missing_in_other:
        raise ValueError(
            f"{name}: {len(missing_in_other):,} reference IDs missing in other dataset."
        )

    if missing_in_reference:
        raise ValueError(
            f"{name}: {len(missing_in_reference):,} other IDs missing in reference dataset."
        )

    other_indexed = other_df.set_index(ID_COLUMN)
    aligned = other_indexed.loc[
        reference_df[ID_COLUMN]
    ].reset_index()

    if not np.array_equal(
        reference_df[ID_COLUMN].values,
        aligned[ID_COLUMN].values,
    ):
        raise ValueError(f"{name}: commit_id alignment failed.")

    print(
        f"[OK] {name}: {len(reference_df):,} rows aligned by commit_id."
    )

    return aligned


def build_llm_features(train_llm, validation_llm, test_llm):
    print_header("ONE-HOT ENCODER")
    print("Encoder will be FIT ON TRAINING DATA ONLY.")

    required = (
        [ID_COLUMN, TARGET_COLUMN]
        + LLM_CATEGORICAL_COLUMNS
        + LLM_NUMERICAL_COLUMNS
    )

    for name, df in [
        ("train", train_llm),
        ("validation", validation_llm),
        ("test", test_llm),
    ]:
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{name}: missing columns: {missing}")

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float32,
    )

    X_train_cat = encoder.fit_transform(
        train_llm[LLM_CATEGORICAL_COLUMNS].astype(str)
    )

    X_validation_cat = encoder.transform(
        validation_llm[LLM_CATEGORICAL_COLUMNS].astype(str)
    )

    X_test_cat = encoder.transform(
        test_llm[LLM_CATEGORICAL_COLUMNS].astype(str)
    )

    categorical_names = (
        encoder.get_feature_names_out(LLM_CATEGORICAL_COLUMNS).tolist()
    )

    print(f"Generated categorical features: {len(categorical_names)}")

    if len(categorical_names) != 37:
        raise ValueError(
            f"Expected 37 one-hot features, got {len(categorical_names)}."
        )

    print("[OK] 37 categorical one-hot features.")

    print("\nCalculating numerical feature medians ONLY from training data.")

    train_num = train_llm[LLM_NUMERICAL_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    validation_num = validation_llm[LLM_NUMERICAL_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    test_num = test_llm[LLM_NUMERICAL_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )

    imputer = SimpleImputer(strategy="median")

    X_train_num = imputer.fit_transform(train_num).astype(np.float32)
    X_validation_num = imputer.transform(validation_num).astype(np.float32)
    X_test_num = imputer.transform(test_num).astype(np.float32)

    X_train_llm = np.hstack([X_train_cat, X_train_num])
    X_validation_llm = np.hstack([X_validation_cat, X_validation_num])
    X_test_llm = np.hstack([X_test_cat, X_test_num])

    feature_names = categorical_names + LLM_NUMERICAL_COLUMNS

    if len(feature_names) != 51:
        raise ValueError(
            f"Expected 51 LLM features, got {len(feature_names)}."
        )

    print("[OK] 37 categorical + 14 numerical = 51 LLM features.")

    return (
        X_train_llm,
        X_validation_llm,
        X_test_llm,
        feature_names,
        encoder,
        imputer,
    )


def evaluate_model(model, X, y, split_name):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)
    mcc = matthews_corrcoef(y, predictions)
    roc_auc = roc_auc_score(y, probabilities)
    pr_auc = average_precision_score(y, probabilities)
    cm = confusion_matrix(y, predictions, labels=[0, 1])

    print_header(f"{split_name.upper()} RESULTS")
    print(f"ACCURACY    : {accuracy:.4f}")
    print(f"PRECISION   : {precision:.4f}")
    print(f"RECALL      : {recall:.4f}")
    print(f"F1          : {f1:.4f}")
    print(f"MCC         : {mcc:.4f}")
    print(f"ROC_AUC     : {roc_auc:.4f}")
    print(f"PR_AUC      : {pr_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "probabilities": probabilities,
        "predictions": predictions,
    }


def clean_metrics(metrics):
    return {
        key: value
        for key, value in metrics.items()
        if key not in ["probabilities", "predictions"]
    }


def plot_roc_pr(y_true, probabilities, model_name, split_name):
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    roc_auc = roc_auc_score(y_true, probabilities)

    plt.figure(figsize=(7, 6))
    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{model_name} (ROC-AUC = {roc_auc:.4f})",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        label="Random classifier",
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} - {split_name.capitalize()} ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    roc_path = (
        PLOTS_DIR
        / f"{model_name.lower().replace(' ', '_')}_"
          f"{split_name.lower()}_roc.png"
    )
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()

    precision, recall, _ = precision_recall_curve(
        y_true, probabilities
    )
    pr_auc = average_precision_score(y_true, probabilities)

    plt.figure(figsize=(7, 6))
    plt.plot(
        recall,
        precision,
        linewidth=2,
        label=f"{model_name} (PR-AUC = {pr_auc:.4f})",
    )

    baseline = np.mean(y_true)
    plt.axhline(
        baseline,
        linestyle="--",
        linewidth=1,
        label=f"Positive prevalence = {baseline:.4f}",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        f"{model_name} - {split_name.capitalize()} "
        "Precision-Recall Curve"
    )
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    pr_path = (
        PLOTS_DIR
        / f"{model_name.lower().replace(' ', '_')}_"
          f"{split_name.lower()}_pr.png"
    )
    plt.savefig(pr_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"{split_name.capitalize()} ROC plot: {roc_path}")
    print(f"{split_name.capitalize()} PR plot:  {pr_path}")


def select_mi_top_k(X_train, X_validation, X_test, y_train, feature_names):
    """
    Fit mutual information ONLY on X_train/y_train.
    Validation and test are only transformed by column selection.
    """

    print_header("MUTUAL INFORMATION FEATURE SELECTION")

    print("Fitting mutual information on TRAINING DATA ONLY.")
    print(f"Full feature space: {X_train.shape[1]}")
    print(f"Selecting top {TOP_K} features.")

    mi_scores = mutual_info_classif(
        X_train,
        y_train,
        random_state=RANDOM_STATE,
        n_neighbors=3,
    )

    if len(mi_scores) != len(feature_names):
        raise ValueError(
            "MI score count does not match feature-name count."
        )

    ranking = pd.DataFrame({
        "feature": feature_names,
        "mi_score": mi_scores,
        "original_index": np.arange(len(feature_names)),
    }).sort_values(
        ["mi_score", "original_index"],
        ascending=[False, True],
    ).reset_index(drop=True)

    ranking["rank"] = np.arange(1, len(ranking) + 1)

    selected = ranking.head(TOP_K).copy()

    selected_indices = selected["original_index"].to_numpy(dtype=int)
    selected_names = selected["feature"].tolist()

    print("\nTop 25 MI-selected features:")
    print(
        selected[
            ["rank", "feature", "mi_score", "original_index"]
        ].to_string(index=False)
    )

    print("\nSelected feature composition:")
    print(
        f"  JIT selected:      "
        f"{sum(name.startswith('jit_') for name in selected_names)}"
    )
    print(
        f"  CodeBERT selected: "
        f"{sum(name.startswith('codebert_') for name in selected_names)}"
    )
    # More explicit LLM count based on the constructed feature-name groups.
    llm_selected = sum(
        name in set(feature_names[-LLM_DIM:])
        for name in selected_names
    )
    print(f"  LLM selected:      {llm_selected}")

    X_train_selected = X_train[:, selected_indices]
    X_validation_selected = X_validation[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]

    print("\nSelected shapes:")
    print(f"X_train:      {X_train_selected.shape}")
    print(f"X_validation: {X_validation_selected.shape}")
    print(f"X_test:       {X_test_selected.shape}")

    if X_train_selected.shape[1] != TOP_K:
        raise ValueError("Selected training matrix does not have 25 features.")
    if X_validation_selected.shape[1] != TOP_K:
        raise ValueError("Selected validation matrix does not have 25 features.")
    if X_test_selected.shape[1] != TOP_K:
        raise ValueError("Selected test matrix does not have 25 features.")

    print("[OK] MI top-25 selection verified.")

    ranking.to_csv(
        METRICS_DIR / "mutual_information_feature_ranking.csv",
        index=False,
    )

    selected.to_csv(
        METRICS_DIR / "selected_top25_features.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "selected_features.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "selection_method": "mutual_information",
                "fit_on": "train_only",
                "full_feature_count": FULL_FEATURES,
                "selected_feature_count": TOP_K,
                "selected_features": selected_names,
                "selected_indices": selected_indices.tolist(),
            },
            f,
            indent=2,
        )

    return (
        X_train_selected,
        X_validation_selected,
        X_test_selected,
        selected,
    )


def main():

    print_header(
        "EXPERIMENT 7D: MI TOP-25 THREE-WAY FEATURE FUSION"
    )

    print(
        """
Methodology:
  Split: Project-wise chronological 70/15/15
  Full feature space: 12 JIT + 384 CodeBERT + 51 LLM = 447
  Feature selection: Mutual Information TOP-25
  MI fitting: TRAIN ONLY
  PCA source representation: 384 components
  PCA fitting: TRAIN ONLY
  One-hot encoding: TRAIN ONLY
  Median imputation: TRAIN ONLY
  Resampling: NOT USED
  Random train/test split: NOT USED
  commit_id: alignment/verification only
  Validation: reporting only
  Test set: final evaluation only
  Threshold: 0.5

Models:
  Random Forest
  XGBoost
  LightGBM
"""
    )

    # ========================================================================
    # CHECK FILES
    # ========================================================================

    print_header("CHECKING INPUT FILES")

    for path in CODEBERT_FILES.values():
        check_file_exists(path)

    for path in LLM_FILES.values():
        check_file_exists(path)

    for path in JIT_FILES.values():
        check_file_exists(path)

    print("[OK] All required files exist.")

    # ========================================================================
    # LOAD DATA
    # ========================================================================

    codebert = {}
    llm = {}
    jit = {}

    for split in ["train", "validation", "test"]:

        print_header(f"LOADING {split.upper()} DATA")

        codebert[split] = pd.read_csv(CODEBERT_FILES[split])
        print(
            f"CodeBERT: {len(codebert[split]):,} rows, "
            f"{len(codebert[split].columns)} columns"
        )
        verify_ids(f"CodeBERT {split}", codebert[split])

        llm[split] = pd.read_csv(LLM_FILES[split])
        print(
            f"LLM:      {len(llm[split]):,} rows, "
            f"{len(llm[split].columns)} columns"
        )
        verify_ids(f"LLM {split}", llm[split])

        jit[split] = pd.read_csv(JIT_FILES[split])
        print(
            f"JIT:      {len(jit[split]):,} rows, "
            f"{len(jit[split].columns)} columns"
        )
        verify_ids(f"JIT {split}", jit[split])

    # ========================================================================
    # SPLIT SIZE CHECK
    # ========================================================================

    print_header("SPLIT VERIFICATION")

    expected_sizes = {
        "train": 41998,
        "validation": 8998,
        "test": 9000,
    }

    for split, expected in expected_sizes.items():
        sizes = [
            len(codebert[split]),
            len(llm[split]),
            len(jit[split]),
        ]

        print(
            f"{split.capitalize():12s}: "
            f"CodeBERT={sizes[0]:,}, "
            f"LLM={sizes[1]:,}, "
            f"JIT={sizes[2]:,}"
        )

        if any(size != expected for size in sizes):
            raise ValueError(f"{split}: unexpected row count.")

    # ========================================================================
    # ALIGN DATASETS
    # ========================================================================

    print_header("COMMIT-ID ALIGNMENT")

    aligned_llm = {}
    aligned_jit = {}

    for split in ["train", "validation", "test"]:

        aligned_llm[split] = align_by_commit_id(
            codebert[split],
            llm[split],
            f"LLM {split}",
        )

        aligned_jit[split] = align_by_commit_id(
            codebert[split],
            jit[split],
            f"JIT {split}",
        )

    # ========================================================================
    # CROSS-SPLIT OVERLAP
    # ========================================================================

    print_header("CROSS-SPLIT COMMIT-ID VERIFICATION")

    verify_no_overlap(
        codebert["train"],
        codebert["validation"],
        codebert["test"],
    )

    # ========================================================================
    # TARGET ALIGNMENT
    # ========================================================================

    print_header("TARGET ALIGNMENT VERIFICATION")

    y = {}

    for split in ["train", "validation", "test"]:

        y_codebert = normalize_target(
            codebert[split][TARGET_COLUMN]
        ).to_numpy()

        y_llm = normalize_target(
            aligned_llm[split][TARGET_COLUMN]
        ).to_numpy()

        y_jit = normalize_target(
            aligned_jit[split][TARGET_COLUMN]
        ).to_numpy()

        if not (
            np.array_equal(y_codebert, y_llm)
            and np.array_equal(y_codebert, y_jit)
        ):
            raise ValueError(
                f"{split}: target labels do not match across all datasets."
            )

        y[split] = y_codebert

        print(
            f"[OK] {split.capitalize()}: "
            "JIT, CodeBERT and LLM labels match."
        )

    # ========================================================================
    # CODEBERT FEATURES
    # ========================================================================

    print_header("PREPARING CODEBERT FEATURES")

    X_codebert = {}

    for split in ["train", "validation", "test"]:
        X_codebert[split] = parse_embedding_column(
            codebert[split]["embedding"],
            CODEBERT_SOURCE_DIM,
        )

        print(
            f"X_{split} CodeBERT: "
            f"{X_codebert[split].shape}"
        )

        if X_codebert[split].shape[1] != CODEBERT_SOURCE_DIM:
            raise ValueError(
                f"{split}: expected {CODEBERT_SOURCE_DIM} CodeBERT features."
            )

    print(
        f"[OK] All CodeBERT representations are "
        f"{CODEBERT_SOURCE_DIM}-D."
    )

    # ========================================================================
    # JIT FEATURES
    # ========================================================================

    print_header("PREPARING JIT FEATURES")

    X_jit = {}

    for split in ["train", "validation", "test"]:

        missing = [
            col
            for col in JIT_FEATURES
            if col not in aligned_jit[split].columns
        ]

        if missing:
            raise ValueError(
                f"JIT {split}: missing features: {missing}"
            )

        X_jit[split] = (
            aligned_jit[split][JIT_FEATURES]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32)
        )

        print(
            f"X_{split} JIT: "
            f"{X_jit[split].shape}"
        )

    # ========================================================================
    # LLM FEATURES
    # ========================================================================

    print_header("PREPARING LLM REASONING FEATURES")

    (
        X_train_llm,
        X_validation_llm,
        X_test_llm,
        llm_feature_names,
        encoder,
        imputer,
    ) = build_llm_features(
        aligned_llm["train"],
        aligned_llm["validation"],
        aligned_llm["test"],
    )

    X_llm = {
        "train": X_train_llm,
        "validation": X_validation_llm,
        "test": X_test_llm,
    }

    # ========================================================================
    # FULL THREE-WAY FUSION
    # ========================================================================

    print_header("FULL THREE-WAY FEATURE FUSION")

    X_full = {}

    for split in ["train", "validation", "test"]:

        X_full[split] = np.hstack([
            X_jit[split],
            X_codebert[split],
            X_llm[split],
        ])

        print(
            f"X_{split}: "
            f"{X_full[split].shape}"
        )

    jit_feature_names = [
        f"jit_{feature}"
        for feature in JIT_FEATURES
    ]

    codebert_feature_names = [
        f"codebert_pca_{i + 1}"
        for i in range(CODEBERT_SOURCE_DIM)
    ]

    feature_names = (
        jit_feature_names
        + codebert_feature_names
        + llm_feature_names
    )

    print(f"\nJIT features       : {len(jit_feature_names)}")
    print(f"CodeBERT features  : {len(codebert_feature_names)}")
    print(f"LLM features       : {len(llm_feature_names)}")
    print(f"Total features     : {len(feature_names)}")

    if len(feature_names) != FULL_FEATURES:
        raise ValueError(
            f"Expected {FULL_FEATURES} features, got {len(feature_names)}."
        )

    for split in ["train", "validation", "test"]:
        if X_full[split].shape[1] != FULL_FEATURES:
            raise ValueError(
                f"{split}: expected {FULL_FEATURES} features, "
                f"got {X_full[split].shape[1]}."
            )

    print("[OK] 12 JIT + 384 CodeBERT + 51 LLM = 447 features.")

    # ========================================================================
    # FEATURE QUALITY
    # ========================================================================

    print_header("FULL FEATURE QUALITY CHECK")

    for split in ["train", "validation", "test"]:

        nan_count = np.isnan(X_full[split]).sum()
        inf_count = np.isinf(X_full[split]).sum()

        print(
            f"{split.capitalize():12s}: "
            f"NaN={nan_count:,}, "
            f"Inf={inf_count:,}"
        )

        if nan_count != 0 or inf_count != 0:
            raise ValueError(
                f"{split}: NaN or infinite values detected."
            )

    print("[OK] No NaN or infinite values remain.")

    # ========================================================================
    # TARGET DISTRIBUTION
    # ========================================================================

    print_header("TARGET DISTRIBUTION")

    for split in ["train", "validation", "test"]:

        positives = int(y[split].sum())
        total = len(y[split])
        negatives = total - positives

        print(
            f"{split.capitalize():12s}: "
            f"buggy={positives:,} / {total:,} "
            f"({positives / total:.2%}), "
            f"non-buggy={negatives:,}"
        )

    # ========================================================================
    # MUTUAL INFORMATION TOP-25 SELECTION
    # ========================================================================

    (
        X_train,
        X_validation,
        X_test,
        mi_ranking,
    ) = select_mi_top_k(
        X_full["train"],
        X_full["validation"],
        X_full["test"],
        y["train"],
        feature_names,
    )

    # Save selected feature matrices for reproducibility.
    np.save(OUTPUT_DIR / "X_train_mi_top25.npy", X_train)
    np.save(OUTPUT_DIR / "X_validation_mi_top25.npy", X_validation)
    np.save(OUTPUT_DIR / "X_test_mi_top25.npy", X_test)

    # ========================================================================
    # CLASS IMBALANCE
    # ========================================================================

    train_positive = int(y["train"].sum())
    train_negative = len(y["train"]) - train_positive

    scale_pos_weight = train_negative / train_positive

    print_header("CLASS IMBALANCE")

    print(f"Training negatives: {train_negative:,}")
    print(f"Training positives: {train_positive:,}")
    print(f"XGBoost scale_pos_weight: {scale_pos_weight:.4f}")
    print(f"LightGBM scale_pos_weight: {scale_pos_weight:.4f}")

    # ========================================================================
    # RANDOM FOREST
    # ========================================================================

    all_results = []
    all_detailed_metrics = {}

    print_header("TRAINING: RANDOM_FOREST")

    rf = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    rf.fit(X_train, y["train"])

    rf_validation = evaluate_model(
        rf,
        X_validation,
        y["validation"],
        "validation",
    )

    rf_test = evaluate_model(
        rf,
        X_test,
        y["test"],
        "test",
    )

    joblib.dump(rf, MODEL_DIR / "random_forest.joblib")

    plot_roc_pr(
        y["validation"],
        rf_validation["probabilities"],
        "Random Forest",
        "validation",
    )

    plot_roc_pr(
        y["test"],
        rf_test["probabilities"],
        "Random Forest",
        "test",
    )

    all_results.append({
        "model": "random_forest",
        "test_accuracy": rf_test["accuracy"],
        "test_precision": rf_test["precision"],
        "test_recall": rf_test["recall"],
        "test_f1": rf_test["f1"],
        "test_mcc": rf_test["mcc"],
        "test_roc_auc": rf_test["roc_auc"],
        "test_pr_auc": rf_test["pr_auc"],
    })

    all_detailed_metrics["random_forest"] = {
        "validation": clean_metrics(rf_validation),
        "test": clean_metrics(rf_test),
    }

    pd.DataFrame({
        ID_COLUMN: codebert["test"][ID_COLUMN].values,
        "actual_buggy": y["test"],
        "predicted_probability": rf_test["probabilities"],
        "predicted_buggy": rf_test["predictions"],
    }).to_csv(
        PREDICTIONS_DIR / "random_forest_test_predictions.csv",
        index=False,
    )

    # ========================================================================
    # XGBOOST
    # ========================================================================

    print_header("TRAINING: XGBOOST")

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    xgb.fit(X_train, y["train"])

    xgb_validation = evaluate_model(
        xgb,
        X_validation,
        y["validation"],
        "validation",
    )

    xgb_test = evaluate_model(
        xgb,
        X_test,
        y["test"],
        "test",
    )

    joblib.dump(xgb, MODEL_DIR / "xgboost.joblib")

    plot_roc_pr(
        y["validation"],
        xgb_validation["probabilities"],
        "XGBoost",
        "validation",
    )

    plot_roc_pr(
        y["test"],
        xgb_test["probabilities"],
        "XGBoost",
        "test",
    )

    all_results.append({
        "model": "xgboost",
        "test_accuracy": xgb_test["accuracy"],
        "test_precision": xgb_test["precision"],
        "test_recall": xgb_test["recall"],
        "test_f1": xgb_test["f1"],
        "test_mcc": xgb_test["mcc"],
        "test_roc_auc": xgb_test["roc_auc"],
        "test_pr_auc": xgb_test["pr_auc"],
    })

    all_detailed_metrics["xgboost"] = {
        "validation": clean_metrics(xgb_validation),
        "test": clean_metrics(xgb_test),
    }

    pd.DataFrame({
        ID_COLUMN: codebert["test"][ID_COLUMN].values,
        "actual_buggy": y["test"],
        "predicted_probability": xgb_test["probabilities"],
        "predicted_buggy": xgb_test["predictions"],
    }).to_csv(
        PREDICTIONS_DIR / "xgboost_test_predictions.csv",
        index=False,
    )

    # ========================================================================
    # LIGHTGBM
    # ========================================================================

    print_header("TRAINING: LIGHTGBM")

    lgbm = LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    lgbm.fit(X_train, y["train"])

    lgbm_validation = evaluate_model(
        lgbm,
        X_validation,
        y["validation"],
        "validation",
    )

    lgbm_test = evaluate_model(
        lgbm,
        X_test,
        y["test"],
        "test",
    )

    joblib.dump(lgbm, MODEL_DIR / "lightgbm.joblib")

    plot_roc_pr(
        y["validation"],
        lgbm_validation["probabilities"],
        "LightGBM",
        "validation",
    )

    plot_roc_pr(
        y["test"],
        lgbm_test["probabilities"],
        "LightGBM",
        "test",
    )

    all_results.append({
        "model": "lightgbm",
        "test_accuracy": lgbm_test["accuracy"],
        "test_precision": lgbm_test["precision"],
        "test_recall": lgbm_test["recall"],
        "test_f1": lgbm_test["f1"],
        "test_mcc": lgbm_test["mcc"],
        "test_roc_auc": lgbm_test["roc_auc"],
        "test_pr_auc": lgbm_test["pr_auc"],
    })

    all_detailed_metrics["lightgbm"] = {
        "validation": clean_metrics(lgbm_validation),
        "test": clean_metrics(lgbm_test),
    }

    pd.DataFrame({
        ID_COLUMN: codebert["test"][ID_COLUMN].values,
        "actual_buggy": y["test"],
        "predicted_probability": lgbm_test["probabilities"],
        "predicted_buggy": lgbm_test["predictions"],
    }).to_csv(
        PREDICTIONS_DIR / "lightgbm_test_predictions.csv",
        index=False,
    )

    # ========================================================================
    # FINAL RESULTS
    # ========================================================================

    results_df = pd.DataFrame(all_results)

    results_path = METRICS_DIR / "experiment_7d_results.csv"
    results_df.to_csv(results_path, index=False)

    metrics_json = {
        "experiment": "experiment_7d_mi_top25_jit_codebert_llm_fusion",
        "methodology": {
            "split": "project-wise chronological 70/15/15",
            "full_features": FULL_FEATURES,
            "selected_features": TOP_K,
            "feature_selection": "mutual information",
            "feature_selection_fit_on": "train only",
            "jit_features": JIT_DIM,
            "codebert_features": CODEBERT_SOURCE_DIM,
            "llm_features": LLM_DIM,
            "pca": "384 components; full 384 used before MI selection",
            "pca_fitted_on": "train only",
            "one_hot_encoding": "train only",
            "median_imputation": "train only",
            "resampling": False,
            "random_split": False,
            "threshold": THRESHOLD,
            "test_used_for_tuning": False,
        },
        "dataset": {
            "train_rows": len(y["train"]),
            "validation_rows": len(y["validation"]),
            "test_rows": len(y["test"]),
            "total_rows": (
                len(y["train"])
                + len(y["validation"])
                + len(y["test"])
            ),
        },
        "class_balance": {
            "train_positive": train_positive,
            "train_negative": train_negative,
            "scale_pos_weight": scale_pos_weight,
        },
        "models": all_detailed_metrics,
    }

    with open(
        METRICS_DIR / "experiment_7d_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics_json, f, indent=2)

    # ========================================================================
    # FEATURE METADATA
    # ========================================================================

    selected_feature_names = mi_ranking.head(TOP_K)["feature"].tolist()

    feature_metadata = {
        "full_feature_count": FULL_FEATURES,
        "selected_feature_count": TOP_K,
        "selection_method": "mutual_information",
        "selection_fit_on": "train_only",
        "selected_features": selected_feature_names,
        "selected_feature_indices": (
            mi_ranking.head(TOP_K)["original_index"]
            .astype(int)
            .tolist()
        ),
        "full_feature_names": feature_names,
    }

    with open(
        OUTPUT_DIR / "feature_names.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(feature_metadata, f, indent=2)

    # ========================================================================
    # SAVE PREPROCESSORS
    # ========================================================================

    joblib.dump(
        encoder,
        OUTPUT_DIR / "one_hot_encoder.joblib",
    )

    joblib.dump(
        imputer,
        OUTPUT_DIR / "numerical_median_imputer.joblib",
    )

    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    configuration = {
        "experiment": "7D",
        "name": "MI Top-25 JIT + CodeBERT + LLM Feature Fusion",
        "feature_counts": {
            "full": FULL_FEATURES,
            "selected": TOP_K,
            "jit": JIT_DIM,
            "codebert": CODEBERT_SOURCE_DIM,
            "llm": LLM_DIM,
        },
        "feature_selection": {
            "method": "mutual_information",
            "fit_on": "train_only",
            "top_k": TOP_K,
            "random_state": RANDOM_STATE,
            "n_neighbors": 3,
        },
        "jit_features": JIT_FEATURES,
        "llm_categorical_columns": LLM_CATEGORICAL_COLUMNS,
        "llm_numerical_columns": LLM_NUMERICAL_COLUMNS,
        "codebert_input": {
            "train": str(CODEBERT_FILES["train"]),
            "validation": str(CODEBERT_FILES["validation"]),
            "test": str(CODEBERT_FILES["test"]),
            "pca_components_source": CODEBERT_SOURCE_DIM,
            "pca_fit": "train_only",
        },
        "llm_input": {
            "train": str(LLM_FILES["train"]),
            "validation": str(LLM_FILES["validation"]),
            "test": str(LLM_FILES["test"]),
            "one_hot_features": 37,
            "numerical_features": 14,
        },
        "jit_input": {
            "train": str(JIT_FILES["train"]),
            "validation": str(JIT_FILES["validation"]),
            "test": str(JIT_FILES["test"]),
        },
        "models": {
            "random_forest": {
                "n_estimators": 300,
                "max_features": "sqrt",
                "class_weight": "balanced",
                "random_state": RANDOM_STATE,
                "n_jobs": -1,
            },
            "xgboost": {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "eval_metric": "logloss",
                "scale_pos_weight": scale_pos_weight,
                "random_state": RANDOM_STATE,
                "n_jobs": -1,
            },
            "lightgbm": {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": scale_pos_weight,
                "objective": "binary",
                "random_state": RANDOM_STATE,
                "n_jobs": -1,
                "verbosity": -1,
            },
        },
        "threshold": THRESHOLD,
        "resampling": False,
        "random_split": False,
        "test_used_for_tuning": False,
    }

    with open(
        OUTPUT_DIR / "experiment_7d_configuration.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(configuration, f, indent=2)

    # ========================================================================
    # FINAL OUTPUT
    # ========================================================================

    print_header("EXPERIMENT 7D COMPLETED")

    print(
        """
Feature configuration:
  Full feature space:  447
  JIT features:        12
  CodeBERT features:   384
  LLM features:        51
  MI-selected:         25

Models:
  Random Forest
  XGBoost
  LightGBM
"""
    )

    print("Dataset:")
    print(f"  Train:       {len(y['train']):,}")
    print(f"  Validation:  {len(y['validation']):,}")
    print(f"  Test:        {len(y['test']):,}")
    print(
        f"  Total:       "
        f"{sum(len(y[s]) for s in ['train', 'validation', 'test']):,}"
    )

    print("\nFinal TEST performance:")
    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print(f"\nResults saved to:\n  {results_path}")
    print(f"\nModels saved to:\n  {MODEL_DIR}")
    print(
        f"\nSelected feature ranking saved to:\n  "
        f"{METRICS_DIR / 'mutual_information_feature_ranking.csv'}"
    )
    print(f"\nPlots saved to:\n  {PLOTS_DIR}")


if __name__ == "__main__":
    main()
