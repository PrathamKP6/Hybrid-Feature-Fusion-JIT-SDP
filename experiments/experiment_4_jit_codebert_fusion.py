"""
Experiment 4: JIT + CodeBERT Feature Fusion
=============================================

Purpose:
    Evaluate whether combining traditional JIT/process metrics
    with semantic CodeBERT representations improves defect
    prediction over either feature family alone.

Feature fusion:
    12 JIT features
    +
    384 PCA-reduced CodeBERT features
    =
    396 total features

Data:
    Project-wise chronological 70/15/15 split.

JIT source:
    results/chronological_splits/

CodeBERT source:
    results/pca/

Important:
    - PCA was fitted on TRAIN ONLY in the previous PCA pipeline.
    - No new PCA is fitted here.
    - No random train/test split.
    - No resampling.
    - No stratified sampling.
    - Train/validation/test are joined using commit_id.
    - Test is used only for final evaluation.
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
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------
# Chronological JIT splits
# ------------------------------------------------------------

JIT_SPLIT_DIR = (
    PROJECT_ROOT
    / "results"
    / "chronological_splits"
)

JIT_TRAIN_PATH = (
    JIT_SPLIT_DIR / "train.csv"
)

JIT_VALIDATION_PATH = (
    JIT_SPLIT_DIR / "validation.csv"
)

JIT_TEST_PATH = (
    JIT_SPLIT_DIR / "test.csv"
)


# ------------------------------------------------------------
# PCA CodeBERT splits
# ------------------------------------------------------------

PCA_DIR = (
    PROJECT_ROOT
    / "results"
    / "pca"
)

PCA_TRAIN_PATH = (
    PCA_DIR / "train_pca384.csv"
)

PCA_VALIDATION_PATH = (
    PCA_DIR / "validation_pca384.csv"
)

PCA_TEST_PATH = (
    PCA_DIR / "test_pca384.csv"
)


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "experiment_4_jit_codebert_fusion"
)

METRICS_DIR = (
    OUTPUT_DIR / "metrics"
)

MODELS_DIR = (
    OUTPUT_DIR / "models"
)

PRED_DIR = (
    OUTPUT_DIR / "predictions"
)

PLOTS_DIR = (
    OUTPUT_DIR / "plots"
)

for directory in [
    OUTPUT_DIR,
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
# FEATURE DEFINITIONS
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

EMBEDDING_COLUMN = "embedding"

EMBEDDING_DIMENSION = 384

TOTAL_FEATURES = (
    len(JIT_FEATURES)
    + EMBEDDING_DIMENSION
)

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

    if isinstance(
        value,
        (bool, np.bool_),
    ):
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

    if value_str in {
        "0",
        "false",
    }:
        return 0

    if value_str in {
        "1",
        "true",
    }:
        return 1

    raise ValueError(
        f"Unexpected buggy value: {value!r}"
    )


# ============================================================
# EMBEDDING PARSER
# ============================================================

def parse_embedding(value):
    """
    Parse one 384-D PCA embedding from the embedding column.
    """

    if pd.isna(value):

        raise ValueError(
            "Encountered missing embedding."
        )

    # Already list / tuple / numpy array
    if isinstance(
        value,
        (
            list,
            tuple,
            np.ndarray,
        ),
    ):

        embedding = list(value)

    # Serialized list
    elif isinstance(value, str):

        value = value.strip()

        try:

            embedding = ast.literal_eval(
                value
            )

        except Exception:

            try:

                embedding = json.loads(
                    value
                )

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
            f"Expected "
            f"{EMBEDDING_DIMENSION}-D embedding, "
            f"got {len(embedding)}"
        )

    if not np.isfinite(
        embedding
    ).all():

        raise ValueError(
            "Embedding contains NaN or "
            "infinite values."
        )

    return embedding


# ============================================================
# LOAD JIT SPLIT
# ============================================================

def load_jit_split(
    path: Path,
    split_name: str,
) -> pd.DataFrame:

    print(
        f"\nLoading JIT {split_name}:"
    )

    print(
        f"  {path}"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing JIT split: {path}"
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

    required = (
        BOOKKEEPING_COLUMNS
        + JIT_FEATURES
        + [TARGET]
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{split_name} is missing required "
            f"JIT columns: {missing}"
        )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    df[TARGET] = df[TARGET].apply(
        normalize_buggy
    )

    if df[TARGET].isna().any():

        raise ValueError(
            f"{split_name} contains missing "
            f"buggy labels."
        )

    df[TARGET] = df[TARGET].astype(int)

    # --------------------------------------------------------
    # JIT numeric conversion
    # --------------------------------------------------------

    for feature in JIT_FEATURES:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

    df[JIT_FEATURES] = (
        df[JIT_FEATURES]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    # --------------------------------------------------------
    # Duplicate commit IDs
    # --------------------------------------------------------

    duplicate_ids = int(
        df["commit_id"].duplicated().sum()
    )

    if duplicate_ids > 0:

        raise ValueError(
            f"{split_name} contains "
            f"{duplicate_ids} duplicate commit IDs."
        )

    return df


# ============================================================
# LOAD PCA CODEBERT SPLIT
# ============================================================

def load_codebert_split(
    path: Path,
    split_name: str,
) -> pd.DataFrame:

    print(
        f"\nLoading CodeBERT {split_name}:"
    )

    print(
        f"  {path}"
    )

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
            f"CodeBERT columns: {missing}"
        )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    df[TARGET] = df[TARGET].apply(
        normalize_buggy
    )

    if df[TARGET].isna().any():

        raise ValueError(
            f"{split_name} contains missing "
            f"buggy labels."
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
    # Keep only required metadata + embedding
    # --------------------------------------------------------

    result = df[
        [
            "commit_id",
            "project",
            "author_date",
            TARGET,
            EMBEDDING_COLUMN,
        ]
    ].copy()

    return result


# ============================================================
# JOIN JIT + CODEBERT
# ============================================================

def create_fused_split(
    jit_df: pd.DataFrame,
    codebert_df: pd.DataFrame,
    split_name: str,
):
    """
    Join JIT and CodeBERT features using commit_id.

    Returns:
        metadata dataframe
        fused feature matrix
        target array
    """

    print(
        f"\nCreating {split_name} fused features..."
    )

    # --------------------------------------------------------
    # Verify same commit set
    # --------------------------------------------------------

    jit_ids = set(
        jit_df["commit_id"]
    )

    codebert_ids = set(
        codebert_df["commit_id"]
    )

    missing_codebert = (
        jit_ids - codebert_ids
    )

    missing_jit = (
        codebert_ids - jit_ids
    )

    if missing_codebert:

        raise ValueError(
            f"{split_name}: "
            f"{len(missing_codebert)} JIT commits "
            f"are missing CodeBERT embeddings."
        )

    if missing_jit:

        raise ValueError(
            f"{split_name}: "
            f"{len(missing_jit)} CodeBERT commits "
            f"are missing JIT features."
        )

    # --------------------------------------------------------
    # Preserve JIT split order
    # --------------------------------------------------------

    jit_base = jit_df[
        [
            "commit_id",
            "project",
            "author_date",
            TARGET,
        ]
        + JIT_FEATURES
    ].copy()

    codebert_base = codebert_df[
        [
            "commit_id",
            EMBEDDING_COLUMN,
        ]
    ].copy()

    # --------------------------------------------------------
    # Merge using commit_id
    # --------------------------------------------------------

    merged = jit_base.merge(
        codebert_base,
        on="commit_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if len(merged) != len(jit_df):

        raise ValueError(
            f"{split_name}: merged row count changed. "
            f"Expected {len(jit_df):,}, "
            f"got {len(merged):,}."
        )

    # --------------------------------------------------------
    # Parse embeddings
    # --------------------------------------------------------

    print(
        "  Parsing 384-D PCA embeddings..."
    )

    embeddings = []

    for index, value in enumerate(
        merged[EMBEDDING_COLUMN]
    ):

        try:

            embeddings.append(
                parse_embedding(value)
            )

        except Exception as exc:

            raise ValueError(
                f"Invalid CodeBERT embedding in "
                f"{split_name} row {index}: {exc}"
            ) from exc

    codebert_matrix = np.vstack(
        embeddings
    ).astype(np.float32)

    # --------------------------------------------------------
    # JIT matrix
    # --------------------------------------------------------

    jit_matrix = merged[
        JIT_FEATURES
    ].to_numpy(
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Training-derived imputation is performed later.
    # --------------------------------------------------------

    fused_matrix = np.hstack(
        [
            jit_matrix,
            codebert_matrix,
        ]
    ).astype(np.float32)

    print(
        f"  JIT dimensions: "
        f"{jit_matrix.shape[1]}"
    )

    print(
        f"  CodeBERT dimensions: "
        f"{codebert_matrix.shape[1]}"
    )

    print(
        f"  Fused dimensions: "
        f"{fused_matrix.shape[1]}"
    )

    if fused_matrix.shape[1] != TOTAL_FEATURES:

        raise ValueError(
            f"Expected {TOTAL_FEATURES} total features, "
            f"got {fused_matrix.shape[1]}."
        )

    metadata = merged[
        [
            "commit_id",
            "project",
            "author_date",
            TARGET,
        ]
    ].copy()

    y = merged[
        TARGET
    ].to_numpy(
        dtype=np.int32
    )

    return (
        metadata,
        fused_matrix,
        y,
    )


# ============================================================
# VERIFY ALL SPLITS
# ============================================================

def verify_splits(
    train_df,
    validation_df,
    test_df,
):

    print(
        "\n" + "=" * 70
    )

    print(
        "SPLIT VERIFICATION"
    )

    print(
        "=" * 70
    )

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
            f"Expected 59,996 total rows, "
            f"got {total:,}."
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

    train_val = (
        train_ids & validation_ids
    )

    train_test = (
        train_ids & test_ids
    )

    validation_test = (
        validation_ids & test_ids
    )

    if train_val:

        raise ValueError(
            f"Train/validation overlap: "
            f"{len(train_val)}"
        )

    if train_test:

        raise ValueError(
            f"Train/test overlap: "
            f"{len(train_test)}"
        )

    if validation_test:

        raise ValueError(
            f"Validation/test overlap: "
            f"{len(validation_test)}"
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

    print(
        "\n" + "=" * 70
    )

    print(
        "TARGET DISTRIBUTION"
    )

    print(
        "=" * 70
    )

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
            buggy_count
            / total
            * 100
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
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    split_name,
):

    probabilities = model.predict_proba(
        X
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = calculate_metrics(
        y,
        predictions,
        probabilities,
    )

    cm = confusion_matrix(
        y,
        predictions,
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

    print(
        "\nConfusion Matrix:"
    )

    print(cm)

    return (
        metrics,
        predictions,
        probabilities,
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
    metadata,
    predictions,
    probabilities,
    model_name,
    split_name,
):

    output = metadata[
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
# ROC + PR CURVES
# ============================================================

def plot_roc_pr(
    y_true,
    probabilities,
    model_name,
    split_name,
):

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    fpr, tpr, _ = roc_curve(
        y_true,
        probabilities,
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
            probabilities,
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

    print(
        "\n" + "=" * 70
    )

    print(
        "EXPERIMENT 4: JIT + CODEBERT FEATURE FUSION"
    )

    print(
        "=" * 70
    )

    print(
        "\nMethodology:"
    )

    print(
        "  Split: Project-wise chronological 70/15/15"
    )

    print(
        "  JIT features: 12"
    )

    print(
        "  CodeBERT features: 384 PCA components"
    )

    print(
        "  Total fused features: 396"
    )

    print(
        "  PCA: Fitted on TRAIN ONLY"
    )

    print(
        "  Fusion: Early feature concatenation"
    )

    print(
        "  Join key: commit_id"
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

    print(
        "=" * 70
    )


    # ========================================================
    # 1. LOAD JIT SPLITS
    # ========================================================

    jit_train = load_jit_split(
        JIT_TRAIN_PATH,
        "train",
    )

    jit_validation = load_jit_split(
        JIT_VALIDATION_PATH,
        "validation",
    )

    jit_test = load_jit_split(
        JIT_TEST_PATH,
        "test",
    )


    # ========================================================
    # 2. LOAD CODEBERT PCA SPLITS
    # ========================================================

    codebert_train = load_codebert_split(
        PCA_TRAIN_PATH,
        "train",
    )

    codebert_validation = load_codebert_split(
        PCA_VALIDATION_PATH,
        "validation",
    )

    codebert_test = load_codebert_split(
        PCA_TEST_PATH,
        "test",
    )


    # ========================================================
    # 3. VERIFY ORIGINAL SPLITS
    # ========================================================

    verify_splits(
        jit_train,
        jit_validation,
        jit_test,
    )


    # ========================================================
    # 4. TARGET DISTRIBUTION
    # ========================================================

    print_target_distribution(
        jit_train,
        jit_validation,
        jit_test,
    )


    # ========================================================
    # 5. VERIFY JIT/CODEBERT ROW COUNTS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "JIT / CODEBERT ALIGNMENT"
    )

    print(
        "=" * 70
    )

    for name, jit_df, codebert_df in [
        (
            "Train",
            jit_train,
            codebert_train,
        ),
        (
            "Validation",
            jit_validation,
            codebert_validation,
        ),
        (
            "Test",
            jit_test,
            codebert_test,
        ),
    ]:

        print(
            f"{name:12s}: "
            f"JIT={len(jit_df):,}, "
            f"CodeBERT={len(codebert_df):,}"
        )

        if len(jit_df) != len(codebert_df):

            raise ValueError(
                f"{name}: JIT and CodeBERT "
                f"row counts differ."
            )


    # ========================================================
    # 6. CREATE FUSED FEATURES
    # ========================================================

    (
        train_metadata,
        X_train,
        y_train,
    ) = create_fused_split(
        jit_train,
        codebert_train,
        "train",
    )

    (
        validation_metadata,
        X_validation,
        y_validation,
    ) = create_fused_split(
        jit_validation,
        codebert_validation,
        "validation",
    )

    (
        test_metadata,
        X_test,
        y_test,
    ) = create_fused_split(
        jit_test,
        codebert_test,
        "test",
    )


    # ========================================================
    # 7. VERIFY FUSED DIMENSIONS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FEATURE DIMENSION VERIFICATION"
    )

    print(
        "=" * 70
    )

    print(
        f"JIT features:       {len(JIT_FEATURES)}"
    )

    print(
        f"CodeBERT features:  {EMBEDDING_DIMENSION}"
    )

    print(
        f"Total features:     {TOTAL_FEATURES}"
    )

    print(
        f"Train matrix:       {X_train.shape}"
    )

    print(
        f"Validation matrix:  {X_validation.shape}"
    )

    print(
        f"Test matrix:        {X_test.shape}"
    )

    expected_shapes = [
        (
            "train",
            X_train,
            len(train_metadata),
        ),
        (
            "validation",
            X_validation,
            len(validation_metadata),
        ),
        (
            "test",
            X_test,
            len(test_metadata),
        ),
    ]

    for name, matrix, rows in expected_shapes:

        if matrix.shape != (
            rows,
            TOTAL_FEATURES,
        ):

            raise ValueError(
                f"{name}: unexpected fused "
                f"matrix shape {matrix.shape}"
            )


    # ========================================================
    # 8. HANDLE MISSING JIT VALUES
    # ========================================================
    #
    # PCA embeddings should contain no NaNs.
    #
    # JIT features may contain missing values.
    #
    # Medians are calculated ONLY from training data.
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "TRAINING-DERIVED IMPUTATION"
    )

    print(
        "=" * 70
    )

    # JIT columns are the first 12 dimensions.
    jit_train_part = X_train[
        :, :len(JIT_FEATURES)
    ]

    jit_validation_part = X_validation[
        :, :len(JIT_FEATURES)
    ]

    jit_test_part = X_test[
        :, :len(JIT_FEATURES)
    ]

    jit_medians = np.nanmedian(
        jit_train_part,
        axis=0,
    )

    # Check completely missing columns
    if np.isnan(
        jit_medians
    ).any():

        bad_features = [
            JIT_FEATURES[index]
            for index, value
            in enumerate(jit_medians)
            if np.isnan(value)
        ]

        raise ValueError(
            "The following JIT features have no "
            f"valid training values: {bad_features}"
        )

    # Impute each JIT column
    for index, median in enumerate(
        jit_medians
    ):

        train_mask = np.isnan(
            X_train[:, index]
        )

        validation_mask = np.isnan(
            X_validation[:, index]
        )

        test_mask = np.isnan(
            X_test[:, index]
        )

        X_train[
            train_mask,
            index
        ] = median

        X_validation[
            validation_mask,
            index
        ] = median

        X_test[
            test_mask,
            index
        ] = median


    # --------------------------------------------------------
    # Final NaN / infinity check
    # --------------------------------------------------------

    for name, matrix in [
        ("Train", X_train),
        ("Validation", X_validation),
        ("Test", X_test),
    ]:

        if not np.isfinite(
            matrix
        ).all():

            raise ValueError(
                f"{name} fused matrix contains "
                f"NaN or infinite values after "
                f"imputation."
            )

    print(
        "Missing-value handling: "
        "TRAIN-DERIVED MEDIANS ONLY"
    )


    # ========================================================
    # 9. CLASS IMBALANCE
    # ========================================================

    negative_count = np.sum(
        y_train == 0
    )

    positive_count = np.sum(
        y_train == 1
    )

    if positive_count == 0:

        raise ValueError(
            "Training set contains no "
            "positive examples."
        )

    scale_pos_weight = (
        negative_count
        / positive_count
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "MODEL CONFIGURATION"
    )

    print(
        "=" * 70
    )

    print(
        f"XGBoost scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )


    # ========================================================
    # 10. MODELS
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
    # 11. TRAIN + EVALUATE
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
            validation_metadata,
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
            test_metadata,
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
            / f"{model_name}_jit_codebert.joblib"
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
    # 12. SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    results_path = (
        METRICS_DIR
        / "experiment_4_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )


    # ========================================================
    # 13. SAVE SUMMARY JSON
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
    # 14. SAVE CONFIGURATION
    # ========================================================

    configuration = {

        "experiment":
            "Experiment 4 - JIT + CodeBERT Fusion",

        "fusion_method":
            "Early feature concatenation",

        "split_method":
            "Project-wise chronological 70/15/15",

        "train_rows":
            int(len(train_metadata)),

        "validation_rows":
            int(len(validation_metadata)),

        "test_rows":
            int(len(test_metadata)),

        "jit_feature_count":
            len(JIT_FEATURES),

        "jit_features":
            JIT_FEATURES,

        "original_codebert_dimension":
            768,

        "pca_codebert_dimension":
            384,

        "total_feature_dimension":
            TOTAL_FEATURES,

        "pca_fitted_on":
            "training split only",

        "join_key":
            "commit_id",

        "jit_features_used":
            True,

        "codebert_features_used":
            True,

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
        / "experiment_4_configuration.json",
        configuration,
    )


    # ========================================================
    # 15. FINAL TEST RESULTS
    # ========================================================

    print(
        "\n\n" + "=" * 70
    )

    print(
        "EXPERIMENT 4 COMPLETED"
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


    # ========================================================
    # 16. COMPARE AGAINST EXISTING EXPERIMENTS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "COMPARISON WITH EXPERIMENTS 1 AND 2"
    )

    print(
        "=" * 70
    )

    # Known results from completed experiments.
    #
    # These are used only for comparison and are NOT used
    # during model training or test evaluation.

    baseline_results = {

        "JIT_only_XGBoost": {
            "f1": 0.422140,
            "mcc": 0.312212,
            "roc_auc": 0.780634,
            "pr_auc": 0.380177,
        },

        "CodeBERT_only_XGBoost": {
            "f1": 0.361934,
            "mcc": 0.218608,
            "roc_auc": 0.713371,
            "pr_auc": 0.278263,
        },
    }


    comparison_rows = []


    for _, row in results_df.iterrows():

        model_name = row["model"]

        comparison_rows.append(
            {
                "model":
                    f"Hybrid_{model_name}",

                "f1":
                    row["test_f1"],

                "mcc":
                    row["test_mcc"],

                "roc_auc":
                    row["test_roc_auc"],

                "pr_auc":
                    row["test_pr_auc"],
            }
        )


    comparison_rows.extend(
        [
            {
                "model":
                    "JIT_only_XGBoost",

                "f1":
                    baseline_results[
                        "JIT_only_XGBoost"
                    ]["f1"],

                "mcc":
                    baseline_results[
                        "JIT_only_XGBoost"
                    ]["mcc"],

                "roc_auc":
                    baseline_results[
                        "JIT_only_XGBoost"
                    ]["roc_auc"],

                "pr_auc":
                    baseline_results[
                        "JIT_only_XGBoost"
                    ]["pr_auc"],
            },

            {
                "model":
                    "CodeBERT_only_XGBoost",

                "f1":
                    baseline_results[
                        "CodeBERT_only_XGBoost"
                    ]["f1"],

                "mcc":
                    baseline_results[
                        "CodeBERT_only_XGBoost"
                    ]["mcc"],

                "roc_auc":
                    baseline_results[
                        "CodeBERT_only_XGBoost"
                    ]["roc_auc"],

                "pr_auc":
                    baseline_results[
                        "CodeBERT_only_XGBoost"
                    ]["pr_auc"],
            },
        ]
    )


    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_path = (
        METRICS_DIR
        / "experiment_1_2_4_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    print(
        comparison_df.to_string(
            index=False
        )
    )


    # ========================================================
    # 17. CALCULATE HYBRID IMPROVEMENT
    # ========================================================

    improvement_rows = []

    for _, row in results_df.iterrows():

        model_name = row["model"]

        jit_baseline = (
            baseline_results[
                "JIT_only_XGBoost"
            ]
        )

        codebert_baseline = (
            baseline_results[
                "CodeBERT_only_XGBoost"
            ]
        )

        improvement_rows.append(
            {
                "hybrid_model":
                    model_name,

                "delta_f1_vs_jit":
                    row["test_f1"]
                    - jit_baseline["f1"],

                "delta_mcc_vs_jit":
                    row["test_mcc"]
                    - jit_baseline["mcc"],

                "delta_roc_auc_vs_jit":
                    row["test_roc_auc"]
                    - jit_baseline["roc_auc"],

                "delta_pr_auc_vs_jit":
                    row["test_pr_auc"]
                    - jit_baseline["pr_auc"],

                "delta_f1_vs_codebert":
                    row["test_f1"]
                    - codebert_baseline["f1"],

                "delta_mcc_vs_codebert":
                    row["test_mcc"]
                    - codebert_baseline["mcc"],

                "delta_roc_auc_vs_codebert":
                    row["test_roc_auc"]
                    - codebert_baseline["roc_auc"],

                "delta_pr_auc_vs_codebert":
                    row["test_pr_auc"]
                    - codebert_baseline["pr_auc"],
            }
        )


    improvement_df = pd.DataFrame(
        improvement_rows
    )

    improvement_path = (
        METRICS_DIR
        / "hybrid_improvement.csv"
    )

    improvement_df.to_csv(
        improvement_path,
        index=False,
    )


    print(
        "\nHybrid improvement:"
    )

    print(
        improvement_df.to_string(
            index=False
        )
    )


    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "OUTPUT FILES"
    )

    print(
        "=" * 70
    )

    print(
        f"Results: "
        f"{results_path}"
    )

    print(
        f"Comparison: "
        f"{comparison_path}"
    )

    print(
        f"Improvement: "
        f"{improvement_path}"
    )

    print(
        f"Models: "
        f"{MODELS_DIR}"
    )

    print(
        f"Predictions: "
        f"{PRED_DIR}"
    )

    print(
        f"Plots: "
        f"{PLOTS_DIR}"
    )

    print(
        "\n" + "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()