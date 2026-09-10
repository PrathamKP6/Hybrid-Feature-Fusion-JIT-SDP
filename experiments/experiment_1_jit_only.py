"""
Experiment 1: Traditional JIT-Only Baseline
=============================================

Purpose:
    Establish a baseline using only traditional JIT/process metrics.

Dataset:
    Project-wise chronological 70/15/15 split.

Features:
    12 traditional JIT features.

Models:
    1. Random Forest
    2. XGBoost

Important:
    - No random train/test split
    - No stratified sampling
    - No resampling
    - No CodeBERT embeddings
    - No PCA
    - Test set is used only for final evaluation
"""

from pathlib import Path
import json
import warnings

import joblib
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
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = PROJECT_ROOT / "results" / "chronological_splits"

OUTPUT_DIR = PROJECT_ROOT / "results" / "experiment_1_jit_only"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_PATH = SPLIT_DIR / "train.csv"
VALIDATION_PATH = SPLIT_DIR / "validation.csv"
TEST_PATH = SPLIT_DIR / "test.csv"


# ============================================================
# JIT FEATURES
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

TARGET = "buggy"


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def load_split(path, split_name):
    """Load and validate one chronological split."""

    print(f"\nLoading {split_name}:")
    print(f"  {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"Missing split file: {path}"
        )

    df = pd.read_csv(path)

    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required = JIT_FEATURES + [TARGET, "commit_id", "project"]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{split_name} is missing required columns: {missing}"
        )

    # --------------------------------------------------------
    # Target normalization
    # --------------------------------------------------------

    def normalize_buggy(value):
        """
        Convert all expected buggy-label representations
        into integer 0/1.
        """

        if pd.isna(value):
            return np.nan

        # Actual boolean values
        if isinstance(value, (bool, np.bool_)):
            return int(value)

        # Numeric values
        if isinstance(value, (int, float, np.integer, np.floating)):
            if value in (0, 1):
                return int(value)

        # String representations
        value_str = str(value).strip().lower()

        if value_str in {"0", "false"}:
            return 0

        if value_str in {"1", "true"}:
            return 1

        raise ValueError(
            f"Unexpected buggy value: {value!r}"
        )


    df[TARGET] = df[TARGET].apply(normalize_buggy)

    # Check for missing labels
    if df[TARGET].isna().any():
        missing_labels = int(df[TARGET].isna().sum())

        raise ValueError(
            f"{split_name} contains "
            f"{missing_labels} missing buggy labels."
        )

    df[TARGET] = df[TARGET].astype(int)

    # Final validation
    unique_targets = set(df[TARGET].unique())

    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            f"Unexpected normalized target values in "
            f"{split_name}: {unique_targets}"
        )

    # --------------------------------------------------------
    # Feature conversion
    # --------------------------------------------------------

    for feature in JIT_FEATURES:
        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    # Replace infinite values
    df[JIT_FEATURES] = df[JIT_FEATURES].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Missing value handling
    # --------------------------------------------------------

    missing_counts = df[JIT_FEATURES].isna().sum()

    total_missing = missing_counts.sum()

    if total_missing > 0:
        print(
            f"  Missing JIT values: {total_missing:,}"
        )
        print(
            "  Filling missing values with training-derived "
            "medians later."
        )

    # --------------------------------------------------------
    # Duplicate IDs
    # --------------------------------------------------------

    duplicate_ids = df["commit_id"].duplicated().sum()

    if duplicate_ids > 0:
        raise ValueError(
            f"{split_name} contains {duplicate_ids} "
            f"duplicate commit IDs."
        )

    return df


def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate classification metrics."""

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "mcc": matthews_corrcoef(
            y_true,
            y_pred
        ),
        "roc_auc": roc_auc_score(
            y_true,
            y_prob
        ),
        "pr_auc": average_precision_score(
            y_true,
            y_prob
        ),
    }

    return metrics


def evaluate_model(model, X, y, split_name):
    """Evaluate a model on one split."""

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = calculate_metrics(
        y,
        predictions,
        probabilities
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    print(f"\n{'=' * 70}")
    print(f"{split_name.upper()} RESULTS")
    print(f"{'=' * 70}")

    for name, value in metrics.items():
        print(
            f"{name.upper():12s}: {value:.4f}"
        )

    print("\nConfusion Matrix:")
    print(cm)

    return metrics, predictions, probabilities, cm


def save_predictions(
    df,
    predictions,
    probabilities,
    model_name,
    split_name
):
    """Save predictions for later analysis."""

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
        OUTPUT_DIR
        / f"{model_name}_{split_name}_predictions.csv"
    )

    output.to_csv(
        output_path,
        index=False
    )

    return output_path


def save_confusion_matrix(cm, model_name, split_name):
    """Save confusion matrix."""

    cm_df = pd.DataFrame(
        cm,
        index=["Actual_0", "Actual_1"],
        columns=["Predicted_0", "Predicted_1"]
    )

    path = (
        OUTPUT_DIR
        / f"{model_name}_{split_name}_confusion_matrix.csv"
    )

    cm_df.to_csv(path)

    return path


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("EXPERIMENT 1: JIT-ONLY BASELINE")
    print("=" * 70)

    print("\nMethodology:")
    print("  Split: Project-wise chronological 70/15/15")
    print("  Features: 12 traditional JIT metrics")
    print("  Embeddings: NOT USED")
    print("  PCA: NOT USED")
    print("  Resampling: NOT USED")
    print("  Random train/test split: NOT USED")
    print("=" * 70)


    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    train_df = load_split(
        TRAIN_PATH,
        "train"
    )

    validation_df = load_split(
        VALIDATION_PATH,
        "validation"
    )

    test_df = load_split(
        TEST_PATH,
        "test"
    )


    # ========================================================
    # 2. VERIFY SPLIT SIZES
    # ========================================================

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

    total_rows = (
        len(train_df)
        + len(validation_df)
        + len(test_df)
    )

    print(
        f"Total:       {total_rows:,}"
    )

    if total_rows != 59996:
        print(
            f"\nWARNING: Expected 59,996 rows, "
            f"found {total_rows:,}."
        )


    # ========================================================
    # 3. VERIFY NO ID OVERLAP
    # ========================================================

    train_ids = set(train_df["commit_id"])
    validation_ids = set(validation_df["commit_id"])
    test_ids = set(test_df["commit_id"])

    overlap_train_val = train_ids & validation_ids
    overlap_train_test = train_ids & test_ids
    overlap_val_test = validation_ids & test_ids

    if (
        overlap_train_val
        or overlap_train_test
        or overlap_val_test
    ):
        raise ValueError(
            "Commit ID overlap detected between splits!"
        )

    print(
        "\nCommit ID overlap: NONE"
    )


    # ========================================================
    # 4. PREPARE FEATURES
    # ========================================================

    X_train = train_df[JIT_FEATURES].copy()
    X_validation = validation_df[JIT_FEATURES].copy()
    X_test = test_df[JIT_FEATURES].copy()

    y_train = train_df[TARGET].values
    y_validation = validation_df[TARGET].values
    y_test = test_df[TARGET].values


    # ========================================================
    # 5. TRAINING-ONLY MEDIAN IMPUTATION
    # ========================================================
    #
    # IMPORTANT:
    # Medians are calculated ONLY from training data.
    # Validation/test are transformed using those medians.
    #
    # This avoids data leakage.
    # ========================================================

    train_medians = X_train.median()

    X_train = X_train.fillna(train_medians)
    X_validation = X_validation.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    # Any column completely missing in train would still
    # contain NaN. Fail explicitly rather than silently
    # introducing leakage.
    if (
        X_train.isna().any().any()
        or X_validation.isna().any().any()
        or X_test.isna().any().any()
    ):
        raise ValueError(
            "NaN values remain after training-derived "
            "median imputation."
        )


    # ========================================================
    # 6. PRINT TARGET DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 70)
    print("TARGET DISTRIBUTION")
    print("=" * 70)

    for name, y in [
        ("Train", y_train),
        ("Validation", y_validation),
        ("Test", y_test),
    ]:

        buggy_count = int(y.sum())
        total = len(y)
        percentage = 100 * buggy_count / total

        print(
            f"{name:12s}: "
            f"buggy={buggy_count:,} / "
            f"{total:,} "
            f"({percentage:.2f}%)"
        )


    # ========================================================
    # 7. MODEL DEFINITIONS
    # ========================================================

    models = {}


    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    models["random_forest"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    # Calculate class imbalance ONLY from training set.
    negative_count = np.sum(y_train == 0)
    positive_count = np.sum(y_train == 1)

    scale_pos_weight = (
        negative_count / positive_count
        if positive_count > 0
        else 1.0
    )

    print("\n" + "=" * 70)
    print("MODEL CONFIGURATION")
    print("=" * 70)

    print(
        f"XGBoost scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    models["xgboost"] = XGBClassifier(
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
    )


    # ========================================================
    # 8. TRAIN + EVALUATE
    # ========================================================

    all_results = []

    for model_name, model in models.items():

        print("\n\n" + "#" * 70)
        print(
            f"TRAINING: {model_name.upper()}"
        )
        print("#" * 70)

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train
        )

        print(
            "Training completed."
        )


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_metrics, val_predictions, val_probabilities, val_cm = (
            evaluate_model(
                model,
                X_validation,
                y_validation,
                "validation"
            )
        )

        val_prediction_path = save_predictions(
            validation_df,
            val_predictions,
            val_probabilities,
            model_name,
            "validation"
        )

        save_confusion_matrix(
            val_cm,
            model_name,
            "validation"
        )


        # ----------------------------------------------------
        # Final Test
        # ----------------------------------------------------

        test_metrics, test_predictions, test_probabilities, test_cm = (
            evaluate_model(
                model,
                X_test,
                y_test,
                "test"
            )
        )

        test_prediction_path = save_predictions(
            test_df,
            test_predictions,
            test_probabilities,
            model_name,
            "test"
        )

        save_confusion_matrix(
            test_cm,
            model_name,
            "test"
        )


        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        result = {
            "model": model_name,

            "validation_accuracy":
                val_metrics["accuracy"],

            "validation_precision":
                val_metrics["precision"],

            "validation_recall":
                val_metrics["recall"],

            "validation_f1":
                val_metrics["f1"],

            "validation_mcc":
                val_metrics["mcc"],

            "validation_roc_auc":
                val_metrics["roc_auc"],

            "validation_pr_auc":
                val_metrics["pr_auc"],

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

        all_results.append(result)


        # ----------------------------------------------------
        # Save model
        # ----------------------------------------------------

        model_path = (
            OUTPUT_DIR
            / f"{model_name}.joblib"
        )

        joblib.dump(
            model,
            model_path
        )

        print(
            f"\nModel saved: {model_path}"
        )

        print(
            f"Validation predictions: "
            f"{val_prediction_path}"
        )

        print(
            f"Test predictions: "
            f"{test_prediction_path}"
        )


    # ========================================================
    # 9. SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    results_path = (
        OUTPUT_DIR
        / "experiment_1_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )


    # ========================================================
    # 10. SAVE CONFIGURATION
    # ========================================================

    configuration = {
        "experiment": "Experiment 1 - JIT Only",

        "split_method":
            "Project-wise chronological 70/15/15",

        "train_rows":
            int(len(train_df)),

        "validation_rows":
            int(len(validation_df)),

        "test_rows":
            int(len(test_df)),

        "features":
            JIT_FEATURES,

        "target":
            TARGET,

        "embeddings_used":
            False,

        "pca_used":
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
            "XGBoost"
        ],
    }

    config_path = (
        OUTPUT_DIR
        / "experiment_1_configuration.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            configuration,
            f,
            indent=4
        )


    # ========================================================
    # 11. FINAL SUMMARY
    # ========================================================

    print("\n\n" + "=" * 70)
    print("EXPERIMENT 1 COMPLETED")
    print("=" * 70)

    print("\nFinal TEST performance:")

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
        results_df[display_columns]
        .to_string(index=False)
    )

    print("\nResults saved to:")
    print(
        f"  {results_path}"
    )

    print("\nOutput directory:")
    print(
        f"  {OUTPUT_DIR}"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()