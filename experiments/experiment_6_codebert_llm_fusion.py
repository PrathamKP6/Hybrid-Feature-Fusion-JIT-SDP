"""
EXPERIMENT 6: CODEBERT + LLM REASONING FEATURE FUSION

Purpose:
    Evaluate whether structured LLM reasoning features provide
    complementary information to CodeBERT semantic representations.

Feature configuration:
    CodeBERT: 384 PCA features
    LLM reasoning: 51 features
    JIT features: 0
    Total: 435 features

Methodology:
    - Project-wise chronological 70/15/15 split
    - PCA fitted on TRAIN ONLY
    - One-hot encoding fitted on TRAIN ONLY
    - Numerical median imputation fitted on TRAIN ONLY
    - No resampling
    - No random train/test split
    - commit_id used ONLY for alignment and verification
    - Test set used only for final evaluation
    - Classification threshold = 0.5
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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

import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CODEBERT_DIR = PROJECT_ROOT / "results" / "pca"
LLM_DIR = PROJECT_ROOT / "data" / "llm_experiment_dataset"

OUTPUT_DIR = PROJECT_ROOT / "results" / "experiment_6_codebert_llm_fusion"

MODEL_DIR = OUTPUT_DIR / "models"
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
PLOTS_DIR = OUTPUT_DIR / "plots"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


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


# ============================================================================
# LLM FEATURE DEFINITIONS
# ============================================================================

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

EXPECTED_LLM_FEATURE_COUNT = 51
EXPECTED_CODEBERT_FEATURE_COUNT = 384
EXPECTED_TOTAL_FEATURE_COUNT = 435

TARGET_COLUMN = "buggy"
ID_COLUMN = "commit_id"

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
        raise FileNotFoundError(
            f"\nRequired file does not exist:\n{path}"
        )


def normalize_target(series):
    """
    Convert common representations of buggy/non-buggy labels to integers.
    """

    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")

        if values.isna().any():
            raise ValueError(
                f"Target column contains {values.isna().sum()} invalid numeric values."
            )

        unique_values = set(values.unique())

        if not unique_values.issubset({0, 1}):
            raise ValueError(
                f"Target column must contain only 0/1 values. "
                f"Found: {sorted(unique_values)}"
            )

        return values.astype(int)

    mapping = {
        "0": 0,
        "1": 1,
        "false": 0,
        "true": 1,
        "non-buggy": 0,
        "buggy": 1,
        "nonbuggy": 0,
        "bug": 1,
        "clean": 0,
    }

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    if normalized.isna().any():
        bad_values = series[normalized.isna()].unique()
        raise ValueError(
            f"Unknown target values found: {bad_values}"
        )

    return normalized.astype(int)


def get_codebert_columns(df):
    """
    Identify the 384-dimensional PCA embedding column.

    The PCA CSVs retain the original 21-column schema, with the
    'embedding' column containing a JSON representation of the
    384-dimensional PCA vector.
    """

    if "embedding" not in df.columns:
        raise ValueError(
            "Expected 'embedding' column in PCA dataset."
        )

    return ["embedding"]


def parse_embedding_column(series, expected_dim=384):
    """
    Convert JSON/list/string representations of PCA embeddings
    into a dense numpy matrix.
    """

    import ast

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
                f"Unsupported embedding type at row {idx}: "
                f"{type(value)}"
            )

        if not isinstance(vector, (list, tuple)):
            raise ValueError(
                f"Embedding at row {idx} is not a list."
            )

        if len(vector) != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch at row {idx}: "
                f"expected {expected_dim}, got {len(vector)}"
            )

        vectors.append(vector)

    matrix = np.asarray(vectors, dtype=np.float32)

    if matrix.shape[1] != expected_dim:
        raise ValueError(
            f"Final CodeBERT matrix has shape {matrix.shape}; "
            f"expected second dimension {expected_dim}."
        )

    return matrix


def verify_ids(name, df):
    if ID_COLUMN not in df.columns:
        raise ValueError(
            f"{name}: '{ID_COLUMN}' column is missing."
        )

    duplicate_count = df[ID_COLUMN].duplicated().sum()

    print(
        f"{name}: rows={len(df):,}, "
        f"duplicate commit IDs={duplicate_count:,}"
    )

    if duplicate_count != 0:
        raise ValueError(
            f"{name} contains duplicate commit IDs."
        )


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
        raise ValueError(
            "Cross-split commit overlap detected."
        )

    print("\n[OK] No cross-split commit overlap.")


def align_datasets(codebert_df, llm_df, split_name):
    """
    Align LLM rows to CodeBERT rows using commit_id.

    This prevents accidental positional alignment.
    """

    verify_ids(f"CodeBERT {split_name}", codebert_df)
    verify_ids(f"LLM {split_name}", llm_df)

    codebert_ids = set(codebert_df[ID_COLUMN])
    llm_ids = set(llm_df[ID_COLUMN])

    missing_in_llm = codebert_ids - llm_ids
    missing_in_codebert = llm_ids - codebert_ids

    if missing_in_llm:
        raise ValueError(
            f"{split_name}: {len(missing_in_llm):,} CodeBERT "
            f"IDs are missing from LLM dataset."
        )

    if missing_in_codebert:
        raise ValueError(
            f"{split_name}: {len(missing_in_codebert):,} LLM "
            f"IDs are missing from CodeBERT dataset."
        )

    # Reindex LLM dataset according to CodeBERT commit order.
    llm_indexed = llm_df.set_index(ID_COLUMN)

    aligned_llm = llm_indexed.loc[
        codebert_df[ID_COLUMN]
    ].reset_index()

    if not np.array_equal(
        codebert_df[ID_COLUMN].values,
        aligned_llm[ID_COLUMN].values,
    ):
        raise ValueError(
            f"{split_name}: commit_id alignment verification failed."
        )

    print(
        f"[OK] {split_name}: "
        f"{len(codebert_df):,} rows aligned by commit_id."
    )

    return aligned_llm


def build_llm_features(train_llm, val_llm, test_llm):

    # ------------------------------------------------------------------------
    # Check required columns
    # ------------------------------------------------------------------------

    required = (
        [ID_COLUMN, TARGET_COLUMN]
        + LLM_CATEGORICAL_COLUMNS
        + LLM_NUMERICAL_COLUMNS
    )

    for split_name, df in [
        ("train", train_llm),
        ("validation", val_llm),
        ("test", test_llm),
    ]:

        missing = [
            col for col in required
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{split_name}: missing required columns: {missing}"
            )

    # ------------------------------------------------------------------------
    # Categorical features
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ONE-HOT ENCODER")
    print("=" * 70)

    print(
        "Encoder will be FIT ON TRAINING DATA ONLY."
    )

    print(
        f"Categorical columns: {len(LLM_CATEGORICAL_COLUMNS)}"
    )

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float32,
    )

    X_train_cat = encoder.fit_transform(
        train_llm[LLM_CATEGORICAL_COLUMNS].astype(str)
    )

    X_val_cat = encoder.transform(
        val_llm[LLM_CATEGORICAL_COLUMNS].astype(str)
    )

    X_test_cat = encoder.transform(
        test_llm[LLM_CATEGORICAL_COLUMNS].astype(str)
    )

    categorical_feature_names = (
        encoder.get_feature_names_out(
            LLM_CATEGORICAL_COLUMNS
        ).tolist()
    )

    print(
        f"Generated categorical features: "
        f"{len(categorical_feature_names)}"
    )

    if len(categorical_feature_names) != 37:
        raise ValueError(
            "Expected exactly 37 one-hot categorical features, "
            f"but generated {len(categorical_feature_names)}."
        )

    print("[OK] 37 categorical one-hot features.")

    # ------------------------------------------------------------------------
    # Numerical features
    # ------------------------------------------------------------------------

    print(
        "\nCalculating numerical feature medians "
        "ONLY from training data."
    )

    train_num = train_llm[
        LLM_NUMERICAL_COLUMNS
    ].apply(pd.to_numeric, errors="coerce")

    val_num = val_llm[
        LLM_NUMERICAL_COLUMNS
    ].apply(pd.to_numeric, errors="coerce")

    test_num = test_llm[
        LLM_NUMERICAL_COLUMNS
    ].apply(pd.to_numeric, errors="coerce")

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_num = imputer.fit_transform(train_num)
    X_val_num = imputer.transform(val_num)
    X_test_num = imputer.transform(test_num)

    X_train_num = X_train_num.astype(np.float32)
    X_val_num = X_val_num.astype(np.float32)
    X_test_num = X_test_num.astype(np.float32)

    # ------------------------------------------------------------------------
    # Combine LLM features
    # ------------------------------------------------------------------------

    X_train_llm = np.hstack(
        [X_train_cat, X_train_num]
    )

    X_val_llm = np.hstack(
        [X_val_cat, X_val_num]
    )

    X_test_llm = np.hstack(
        [X_test_cat, X_test_num]
    )

    llm_feature_names = (
        categorical_feature_names
        + LLM_NUMERICAL_COLUMNS
    )

    print(
        f"\nFinal LLM feature count: "
        f"{len(llm_feature_names)}"
    )

    if len(llm_feature_names) != EXPECTED_LLM_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_LLM_FEATURE_COUNT} LLM features, "
            f"got {len(llm_feature_names)}."
        )

    print(
        "[OK] 37 categorical + 14 numerical = 51 features."
    )

    return (
        X_train_llm,
        X_val_llm,
        X_test_llm,
        llm_feature_names,
        encoder,
        imputer,
    )


def evaluate_model(model, X, y, split_name):

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(
        y, predictions, zero_division=0
    )
    recall = recall_score(
        y, predictions, zero_division=0
    )
    f1 = f1_score(
        y, predictions, zero_division=0
    )
    mcc = matthews_corrcoef(
        y, predictions
    )
    roc_auc = roc_auc_score(
        y, probabilities
    )
    pr_auc = average_precision_score(
        y, probabilities
    )

    cm = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    )

    print("\n" + "=" * 70)
    print(f"{split_name.upper()} RESULTS")
    print("=" * 70)

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


# ============================================================================
# ROC / PR PLOTTING
# ============================================================================

def plot_roc_pr(
    y_true,
    probabilities,
    model_name,
    split_name,
):
    """
    Save ROC and Precision-Recall curves.

    PR-AUC is reported using average_precision_score so the plotted
    value matches the experiment's reported PR-AUC metric.
    """
    safe_model_name = model_name.replace(" ", "_").lower()
    safe_split_name = split_name.lower()

    # ------------------------------------------------------------------------
    # ROC curve
    # ------------------------------------------------------------------------
    fpr, tpr, _ = roc_curve(
        y_true,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    plt.figure(figsize=(7, 6))
    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (ROC-AUC = {roc_auc:.4f})",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        f"{model_name} — {split_name} ROC Curve"
    )
    plt.legend(loc="lower right")
    plt.tight_layout()

    roc_path = (
        PLOTS_DIR
        / f"{safe_model_name}_{safe_split_name}_roc.png"
    )

    plt.savefig(
        roc_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # ------------------------------------------------------------------------
    # Precision-Recall curve
    # ------------------------------------------------------------------------
    precision, recall, _ = precision_recall_curve(
        y_true,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    plt.figure(figsize=(7, 6))
    plt.plot(
        recall,
        precision,
        label=f"{model_name} (PR-AUC = {pr_auc:.4f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        f"{model_name} — {split_name} Precision-Recall Curve"
    )
    plt.legend(loc="lower left")
    plt.tight_layout()

    pr_path = (
        PLOTS_DIR
        / f"{safe_model_name}_{safe_split_name}_pr.png"
    )

    plt.savefig(
        pr_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return roc_path, pr_path


# ============================================================================
# MAIN
# ============================================================================

def main():

    print_header(
        "EXPERIMENT 6: CODEBERT + LLM REASONING FEATURE FUSION"
    )

    print(
        """
Methodology:
  Split: Project-wise chronological 70/15/15
  JIT features: 0
  CodeBERT features: 384
  LLM reasoning features: 51
  Total model features: 435
  PCA: 384 components
  PCA fitting: TRAIN ONLY
  One-hot encoding: TRAIN ONLY
  Median imputation: TRAIN ONLY
  Resampling: NOT USED
  Random train/test split: NOT USED
  commit_id: alignment/verification only
  Test set: final evaluation only
  Threshold: 0.5
"""
    )

    # ========================================================================
    # CHECK INPUT FILES
    # ========================================================================

    print_header("CHECKING INPUT FILES")

    for path in CODEBERT_FILES.values():
        check_file_exists(path)

    for path in LLM_FILES.values():
        check_file_exists(path)

    print("[OK] All required files exist.")

    # ========================================================================
    # LOAD CODEBERT DATA
    # ========================================================================

    codebert = {}

    for split in ["train", "validation", "test"]:

        print_header(
            f"LOADING CODEBERT {split.upper()} DATA"
        )

        path = CODEBERT_FILES[split]

        print(f"Path: {path}")

        df = pd.read_csv(path)

        print(f"Rows:    {len(df):,}")
        print(f"Columns: {len(df.columns):,}")

        verify_ids(
            f"CodeBERT {split}",
            df,
        )

        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"CodeBERT {split}: '{TARGET_COLUMN}' "
                "column missing."
            )

        if "embedding" not in df.columns:
            raise ValueError(
                f"CodeBERT {split}: 'embedding' column missing."
            )

        codebert[split] = df

    # ========================================================================
    # LOAD LLM DATA
    # ========================================================================

    llm = {}

    for split in ["train", "validation", "test"]:

        print_header(
            f"LOADING LLM {split.upper()} DATA"
        )

        path = LLM_FILES[split]

        print(f"Path: {path}")

        df = pd.read_csv(path)

        print(f"Rows:    {len(df):,}")
        print(f"Columns: {len(df.columns):,}")

        verify_ids(
            f"LLM {split}",
            df,
        )

        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"LLM {split}: '{TARGET_COLUMN}' "
                "column missing."
            )

        llm[split] = df

    # ========================================================================
    # VERIFY SPLIT SIZES
    # ========================================================================

    print_header("SPLIT VERIFICATION")

    expected_sizes = {
        "train": 41998,
        "validation": 8998,
        "test": 9000,
    }

    for split, expected in expected_sizes.items():

        cb_rows = len(codebert[split])
        llm_rows = len(llm[split])

        print(
            f"{split.capitalize():12s}: "
            f"CodeBERT={cb_rows:,}, "
            f"LLM={llm_rows:,}"
        )

        if cb_rows != expected:
            raise ValueError(
                f"Unexpected CodeBERT {split} size: "
                f"{cb_rows}, expected {expected}"
            )

        if llm_rows != expected:
            raise ValueError(
                f"Unexpected LLM {split} size: "
                f"{llm_rows}, expected {expected}"
            )

    # ========================================================================
    # ALIGN LLM WITH CODEBERT
    # ========================================================================

    print_header(
        "COMMIT-ID ALIGNMENT"
    )

    aligned_llm = {}

    for split in ["train", "validation", "test"]:

        aligned_llm[split] = align_datasets(
            codebert[split],
            llm[split],
            split,
        )

    # ========================================================================
    # VERIFY CROSS-SPLIT OVERLAP
    # ========================================================================

    print_header(
        "CROSS-SPLIT COMMIT-ID VERIFICATION"
    )

    verify_no_overlap(
        codebert["train"],
        codebert["validation"],
        codebert["test"],
    )

    # Also verify LLM splits independently.
    verify_no_overlap(
        aligned_llm["train"],
        aligned_llm["validation"],
        aligned_llm["test"],
    )

    # ========================================================================
    # VERIFY TARGET ALIGNMENT
    # ========================================================================

    print_header(
        "TARGET ALIGNMENT VERIFICATION"
    )

    for split in ["train", "validation", "test"]:

        cb_target = normalize_target(
            codebert[split][TARGET_COLUMN]
        ).to_numpy()

        llm_target = normalize_target(
            aligned_llm[split][TARGET_COLUMN]
        ).to_numpy()

        if not np.array_equal(
            cb_target,
            llm_target,
        ):
            raise ValueError(
                f"{split}: CodeBERT and LLM target labels "
                "do not match."
            )

        print(
            f"[OK] {split.capitalize()}: "
            "CodeBERT and LLM labels match."
        )

    # ========================================================================
    # BUILD CODEBERT MATRICES
    # ========================================================================

    print_header(
        "PREPARING CODEBERT FEATURES"
    )

    X_train_codebert = parse_embedding_column(
        codebert["train"]["embedding"],
        EXPECTED_CODEBERT_FEATURE_COUNT,
    )

    X_val_codebert = parse_embedding_column(
        codebert["validation"]["embedding"],
        EXPECTED_CODEBERT_FEATURE_COUNT,
    )

    X_test_codebert = parse_embedding_column(
        codebert["test"]["embedding"],
        EXPECTED_CODEBERT_FEATURE_COUNT,
    )

    print(
        f"X_train CodeBERT shape: "
        f"{X_train_codebert.shape}"
    )

    print(
        f"X_validation CodeBERT shape: "
        f"{X_val_codebert.shape}"
    )

    print(
        f"X_test CodeBERT shape: "
        f"{X_test_codebert.shape}"
    )

    # ========================================================================
    # BUILD LLM MATRICES
    # ========================================================================

    print_header(
        "PREPARING LLM REASONING FEATURES"
    )

    (
        X_train_llm,
        X_val_llm,
        X_test_llm,
        llm_feature_names,
        encoder,
        imputer,
    ) = build_llm_features(
        aligned_llm["train"],
        aligned_llm["validation"],
        aligned_llm["test"],
    )

    print(
        f"X_train LLM shape: "
        f"{X_train_llm.shape}"
    )

    print(
        f"X_validation LLM shape: "
        f"{X_val_llm.shape}"
    )

    print(
        f"X_test LLM shape: "
        f"{X_test_llm.shape}"
    )

    # ========================================================================
    # COMBINE CODEBERT + LLM
    # ========================================================================

    print_header(
        "FEATURE FUSION"
    )

    X_train = np.hstack(
        [X_train_codebert, X_train_llm]
    )

    X_validation = np.hstack(
        [X_val_codebert, X_val_llm]
    )

    X_test = np.hstack(
        [X_test_codebert, X_test_llm]
    )

    codebert_feature_names = [
        f"codebert_pca_{i + 1}"
        for i in range(EXPECTED_CODEBERT_FEATURE_COUNT)
    ]

    feature_names = (
        codebert_feature_names
        + llm_feature_names
    )

    print(
        f"CodeBERT features : "
        f"{len(codebert_feature_names)}"
    )

    print(
        f"LLM features      : "
        f"{len(llm_feature_names)}"
    )

    print(
        f"Total features    : "
        f"{len(feature_names)}"
    )

    if len(feature_names) != EXPECTED_TOTAL_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_FEATURE_COUNT} "
            f"total features, got {len(feature_names)}."
        )

    if X_train.shape[1] != EXPECTED_TOTAL_FEATURE_COUNT:
        raise ValueError(
            f"X_train has {X_train.shape[1]} features; "
            f"expected {EXPECTED_TOTAL_FEATURE_COUNT}."
        )

    if X_validation.shape[1] != EXPECTED_TOTAL_FEATURE_COUNT:
        raise ValueError(
            "Validation feature count mismatch."
        )

    if X_test.shape[1] != EXPECTED_TOTAL_FEATURE_COUNT:
        raise ValueError(
            "Test feature count mismatch."
        )

    print(
        "\n[OK] 384 CodeBERT + 51 LLM = 435 features."
    )

    # ========================================================================
    # CHECK NaN / INF
    # ========================================================================

    print_header(
        "FEATURE QUALITY CHECK"
    )

    for name, X in [
        ("Train", X_train),
        ("Validation", X_validation),
        ("Test", X_test),
    ]:

        nan_count = np.isnan(X).sum()
        inf_count = np.isinf(X).sum()

        print(
            f"{name:12s}: "
            f"NaN={nan_count:,}, "
            f"Inf={inf_count:,}"
        )

        if nan_count != 0 or inf_count != 0:
            raise ValueError(
                f"{name} contains NaN or infinite values."
            )

    print(
        "\n[OK] No NaN or infinite values remain."
    )

    # ========================================================================
    # TARGET
    # ========================================================================

    y_train = normalize_target(
        codebert["train"][TARGET_COLUMN]
    ).to_numpy()

    y_validation = normalize_target(
        codebert["validation"][TARGET_COLUMN]
    ).to_numpy()

    y_test = normalize_target(
        codebert["test"][TARGET_COLUMN]
    ).to_numpy()

    # ========================================================================
    # TARGET DISTRIBUTION
    # ========================================================================

    print_header(
        "TARGET DISTRIBUTION"
    )

    for name, y in [
        ("Train", y_train),
        ("Validation", y_validation),
        ("Test", y_test),
    ]:

        positives = int(y.sum())
        total = len(y)
        negatives = total - positives
        prevalence = positives / total

        print(
            f"{name:12s}: "
            f"buggy={positives:,} / {total:,} "
            f"({prevalence:.2%}), "
            f"non-buggy={negatives:,}"
        )

    # ========================================================================
    # CLASS IMBALANCE
    # ========================================================================

    train_positive = int(y_train.sum())
    train_negative = len(y_train) - train_positive

    scale_pos_weight = (
        train_negative / train_positive
    )

    print_header(
        "CLASS IMBALANCE"
    )

    print(
        f"Training negatives: {train_negative:,}"
    )

    print(
        f"Training positives: {train_positive:,}"
    )

    print(
        f"XGBoost scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    # ========================================================================
    # MODEL CONFIGURATION
    # ========================================================================

    print_header(
        "MODEL CONFIGURATION"
    )

    print("Random Forest: configured")
    print("XGBoost: configured")

    # ========================================================================
    # RANDOM FOREST
    # ========================================================================

    print_header(
        "TRAINING: RANDOM_FOREST"
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("Training model...")

    rf.fit(
        X_train,
        y_train,
    )

    print("Training completed.")

    rf_validation = evaluate_model(
        rf,
        X_validation,
        y_validation,
        "validation",
    )

    rf_test = evaluate_model(
        rf,
        X_test,
        y_test,
        "test",
    )

    rf_model_path = (
        MODEL_DIR / "random_forest.joblib"
    )

    joblib.dump(
        rf,
        rf_model_path,
    )

    print(
        f"\nModel saved:\n  {rf_model_path}"
    )

    # ========================================================================
    # SAVE RF PREDICTIONS
    # ========================================================================

    rf_predictions = pd.DataFrame({
        ID_COLUMN: codebert["test"][ID_COLUMN].values,
        "actual_buggy": y_test,
        "predicted_probability": rf_test["probabilities"],
        "predicted_buggy": rf_test["predictions"],
    })

    rf_predictions.to_csv(
        PREDICTIONS_DIR / "random_forest_test_predictions.csv",
        index=False,
    )

    # ========================================================================
    # ROC / PR PLOTS — RANDOM FOREST
    # ========================================================================

    print_header("GENERATING RANDOM FOREST ROC / PR PLOTS")

    rf_val_roc_path, rf_val_pr_path = plot_roc_pr(
        y_validation,
        rf_validation["probabilities"],
        "Random Forest",
        "validation",
    )

    rf_test_roc_path, rf_test_pr_path = plot_roc_pr(
        y_test,
        rf_test["probabilities"],
        "Random Forest",
        "test",
    )

    print(f"Validation ROC plot: {rf_val_roc_path}")
    print(f"Validation PR plot:  {rf_val_pr_path}")
    print(f"Test ROC plot:       {rf_test_roc_path}")
    print(f"Test PR plot:        {rf_test_pr_path}")

    # ========================================================================
    # XGBOOST
    # ========================================================================

    print_header(
        "TRAINING: XGBOOST"
    )

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

    print("Training model...")

    xgb.fit(
        X_train,
        y_train,
    )

    print("Training completed.")

    xgb_validation = evaluate_model(
        xgb,
        X_validation,
        y_validation,
        "validation",
    )

    xgb_test = evaluate_model(
        xgb,
        X_test,
        y_test,
        "test",
    )

    xgb_model_path = (
        MODEL_DIR / "xgboost.joblib"
    )

    joblib.dump(
        xgb,
        xgb_model_path,
    )

    print(
        f"\nModel saved:\n  {xgb_model_path}"
    )

    # ========================================================================
    # SAVE XGB PREDICTIONS
    # ========================================================================

    xgb_predictions = pd.DataFrame({
        ID_COLUMN: codebert["test"][ID_COLUMN].values,
        "actual_buggy": y_test,
        "predicted_probability": xgb_test["probabilities"],
        "predicted_buggy": xgb_test["predictions"],
    })

    xgb_predictions.to_csv(
        PREDICTIONS_DIR / "xgboost_test_predictions.csv",
        index=False,
    )

    # ========================================================================
    # ROC / PR PLOTS — XGBOOST
    # ========================================================================

    print_header("GENERATING XGBOOST ROC / PR PLOTS")

    xgb_val_roc_path, xgb_val_pr_path = plot_roc_pr(
        y_validation,
        xgb_validation["probabilities"],
        "XGBoost",
        "validation",
    )

    xgb_test_roc_path, xgb_test_pr_path = plot_roc_pr(
        y_test,
        xgb_test["probabilities"],
        "XGBoost",
        "test",
    )

    print(f"Validation ROC plot: {xgb_val_roc_path}")
    print(f"Validation PR plot:  {xgb_val_pr_path}")
    print(f"Test ROC plot:       {xgb_test_roc_path}")
    print(f"Test PR plot:        {xgb_test_pr_path}")

    # ========================================================================
    # RESULTS TABLE
    # ========================================================================

    results = pd.DataFrame([
        {
            "model": "random_forest",
            "test_accuracy": rf_test["accuracy"],
            "test_precision": rf_test["precision"],
            "test_recall": rf_test["recall"],
            "test_f1": rf_test["f1"],
            "test_mcc": rf_test["mcc"],
            "test_roc_auc": rf_test["roc_auc"],
            "test_pr_auc": rf_test["pr_auc"],
        },
        {
            "model": "xgboost",
            "test_accuracy": xgb_test["accuracy"],
            "test_precision": xgb_test["precision"],
            "test_recall": xgb_test["recall"],
            "test_f1": xgb_test["f1"],
            "test_mcc": xgb_test["mcc"],
            "test_roc_auc": xgb_test["roc_auc"],
            "test_pr_auc": xgb_test["pr_auc"],
        },
    ])

    results_path = (
        METRICS_DIR / "experiment_6_results.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    # ========================================================================
    # COMPLETE METRICS JSON
    # ========================================================================

    metrics_json = {
        "experiment": "experiment_6_codebert_llm_fusion",
        "methodology": {
            "split": "project-wise chronological 70/15/15",
            "jit_features": 0,
            "codebert_features": 384,
            "llm_features": 51,
            "total_features": 435,
            "pca": "384 components",
            "pca_fitted_on": "train only",
            "one_hot_encoding": "train only",
            "median_imputation": "train only",
            "resampling": False,
            "random_split": False,
            "threshold": THRESHOLD,
            "test_used_for_tuning": False,
        },
        "dataset": {
            "train_rows": len(y_train),
            "validation_rows": len(y_validation),
            "test_rows": len(y_test),
            "total_rows": (
                len(y_train)
                + len(y_validation)
                + len(y_test)
            ),
        },
        "target_distribution": {
            "train": {
                "positive": int(y_train.sum()),
                "negative": int(
                    len(y_train) - y_train.sum()
                ),
            },
            "validation": {
                "positive": int(y_validation.sum()),
                "negative": int(
                    len(y_validation)
                    - y_validation.sum()
                ),
            },
            "test": {
                "positive": int(y_test.sum()),
                "negative": int(
                    len(y_test) - y_test.sum()
                ),
            },
        },
        "random_forest": {
            "validation": {
                k: v
                for k, v in rf_validation.items()
                if k not in ["probabilities", "predictions"]
            },
            "test": {
                k: v
                for k, v in rf_test.items()
                if k not in ["probabilities", "predictions"]
            },
        },
        "xgboost": {
            "validation": {
                k: v
                for k, v in xgb_validation.items()
                if k not in ["probabilities", "predictions"]
            },
            "test": {
                k: v
                for k, v in xgb_test.items()
                if k not in ["probabilities", "predictions"]
            },
        },
    }

    with open(
        METRICS_DIR / "experiment_6_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics_json,
            f,
            indent=2,
        )

    # ========================================================================
    # SAVE FEATURE NAMES
    # ========================================================================

    feature_metadata = {
        "total_features": len(feature_names),
        "codebert_features": EXPECTED_CODEBERT_FEATURE_COUNT,
        "llm_features": EXPECTED_LLM_FEATURE_COUNT,
        "feature_names": feature_names,
        "codebert_feature_names": codebert_feature_names,
        "llm_feature_names": llm_feature_names,
    }

    with open(
        OUTPUT_DIR / "feature_names.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            feature_metadata,
            f,
            indent=2,
        )

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
    # SAVE CONFIGURATION
    # ========================================================================

    configuration = {
        "experiment": 6,
        "name": "CodeBERT + LLM Reasoning Feature Fusion",
        "project_root": str(PROJECT_ROOT),
        "codebert_input": {
            "train": str(CODEBERT_FILES["train"]),
            "validation": str(CODEBERT_FILES["validation"]),
            "test": str(CODEBERT_FILES["test"]),
            "dimensions": 384,
            "pca_fit": "train_only",
        },
        "llm_input": {
            "train": str(LLM_FILES["train"]),
            "validation": str(LLM_FILES["validation"]),
            "test": str(LLM_FILES["test"]),
            "categorical_columns": LLM_CATEGORICAL_COLUMNS,
            "numerical_columns": LLM_NUMERICAL_COLUMNS,
            "one_hot_features": 37,
            "numerical_features": 14,
            "total_features": 51,
        },
        "fusion": {
            "codebert_features": 384,
            "llm_features": 51,
            "total_features": 435,
            "jit_features": 0,
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
        },
        "threshold": THRESHOLD,
        "resampling": False,
        "random_split": False,
        "test_used_for_tuning": False,
    }

    with open(
        OUTPUT_DIR / "experiment_6_configuration.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            configuration,
            f,
            indent=2,
        )

    # ========================================================================
    # FINAL OUTPUT
    # ========================================================================

    print_header(
        "EXPERIMENT 6 COMPLETED"
    )

    print(
        """
Feature configuration:
  CodeBERT PCA features: 384
  LLM categorical features: 37
  LLM numerical features: 14
  Total LLM features: 51
  Total fusion features: 435
"""
    )

    print(
        "Dataset:"
    )

    print(
        f"  Train:       {len(y_train):,}"
    )

    print(
        f"  Validation:  {len(y_validation):,}"
    )

    print(
        f"  Test:        {len(y_test):,}"
    )

    print(
        f"  Total:       "
        f"{len(y_train) + len(y_validation) + len(y_test):,}"
    )

    print("\nFinal TEST performance:")

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print(
        f"\nResults saved to:\n  {results_path}"
    )

    print(
        "\nOne-hot encoder saved to:"
        f"\n  {OUTPUT_DIR / 'one_hot_encoder.joblib'}"
    )

    print(
        "\nMedian imputer saved to:"
        f"\n  {OUTPUT_DIR / 'numerical_median_imputer.joblib'}"
    )

    print(
        "\nFeature names saved to:"
        f"\n  {OUTPUT_DIR / 'feature_names.json'}"
    )

    print(
        "\nConfiguration saved to:"
        f"\n  {OUTPUT_DIR / 'experiment_6_configuration.json'}"
    )

    print(
        "\nPlots saved to:"
        f"\n  {PLOTS_DIR}"
    )


if __name__ == "__main__":
    main()