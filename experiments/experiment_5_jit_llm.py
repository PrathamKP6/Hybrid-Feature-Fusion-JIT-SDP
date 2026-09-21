"""
Experiment 5: JIT + LLM Feature Fusion
======================================

Purpose:
    Evaluate software defect prediction using the combination of:

        1. Traditional JIT/process metrics
        2. LLM-derived semantic reasoning features

Dataset:
    Existing project-wise chronological 70/15/15 split.

Input:
    results/experiment_5_jit_llm/datasets/
        train_jit_llm.csv
        validation_jit_llm.csv
        test_jit_llm.csv

Features:
    12 traditional JIT features
    +
    51 LLM features
    =
    63 total model features

Models:
    1. Random Forest
    2. XGBoost

Evaluation:
    Accuracy
    Precision
    Recall
    F1
    MCC
    ROC-AUC
    PR-AUC
    Confusion Matrix

Important:
    - No random train/test split
    - No resampling
    - No additional PCA
    - No feature selection using validation/test
    - Test set is used only for final evaluation
    - Missing-value medians are calculated from training data only
    - buggy is never included in the feature matrix
    - Metadata is never included in the feature matrix
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
    roc_curve,
    precision_recall_curve,
)

import matplotlib.pyplot as plt
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "results"
    / "experiment_5_jit_llm"
    / "datasets"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "experiment_5_jit_llm"
)

MODEL_DIR = OUTPUT_DIR / "models"

PREDICTION_DIR = OUTPUT_DIR / "predictions"

CONFUSION_DIR = OUTPUT_DIR / "confusion_matrices"

PLOTS_DIR = OUTPUT_DIR / "plots"

METRICS_DIR = OUTPUT_DIR / "metrics"

for directory in [
    OUTPUT_DIR,
    MODEL_DIR,
    PREDICTION_DIR,
    CONFUSION_DIR,
    PLOTS_DIR,
    METRICS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


TRAIN_PATH = DATASET_DIR / "train_jit_llm.csv"
VALIDATION_PATH = DATASET_DIR / "validation_jit_llm.csv"
TEST_PATH = DATASET_DIR / "test_jit_llm.csv"


# ============================================================
# FEATURES
# ============================================================

# These MUST match Experiment 1 exactly.
JIT_FEATURES = [
    "la",
    "ld",
    "nf",
    "nd",
    "ns",
    "ent",
    "ndev",
    "age",
    "nuc",
    "aexp",
    "arexp",
    "asexp",
]


# 37 one-hot categorical features
LLM_CATEGORICAL_FEATURES = [
    "intent_BF",
    "intent_BL",
    "intent_DC",
    "intent_FT",
    "intent_OT",
    "intent_PF",
    "intent_RF",
    "intent_SC",
    "intent_TS",

    "change_AP",
    "change_BL",
    "change_CF",
    "change_DA",
    "change_DC",
    "change_DP",
    "change_LG",
    "change_OT",
    "change_TS",
    "change_UI",

    "risk_H",
    "risk_L",
    "risk_M",

    "complexity_H",
    "complexity_L",
    "complexity_M",

    "scope_C",
    "scope_F",
    "scope_L",
    "scope_M",

    "test_H",
    "test_L",
    "test_M",
    "test_N",

    "security_H",
    "security_L",
    "security_M",
    "security_N",
]


# 14 numerical confidence/margin features
LLM_NUMERICAL_FEATURES = [
    "intent_confidence",
    "intent_margin",

    "change_confidence",
    "change_margin",

    "risk_confidence",
    "risk_margin",

    "complexity_confidence",
    "complexity_margin",

    "scope_confidence",
    "scope_margin",

    "test_confidence",
    "test_margin",

    "security_confidence",
    "security_margin",
]


LLM_FEATURES = (
    LLM_CATEGORICAL_FEATURES
    + LLM_NUMERICAL_FEATURES
)


ALL_FEATURES = JIT_FEATURES + LLM_FEATURES

TARGET = "buggy"

EXPECTED_JIT_FEATURE_COUNT = 12
EXPECTED_LLM_FEATURE_COUNT = 51
EXPECTED_TOTAL_FEATURE_COUNT = 63


# ============================================================
# EXPECTED ROW COUNTS
# ============================================================

EXPECTED_ROWS = {
    "train": 41998,
    "validation": 8998,
    "test": 9000,
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

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
    if isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
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


def load_split(path, split_name):
    """
    Load and validate one prepared JIT + LLM split.
    """

    print("\n" + "=" * 70)
    print(f"LOADING {split_name.upper()} DATA")
    print("=" * 70)

    print(f"Path: {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset file: {path}"
        )

    df = pd.read_csv(path)

    print(
        f"Rows:    {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Row count verification
    # --------------------------------------------------------

    expected_rows = EXPECTED_ROWS[split_name]

    if len(df) != expected_rows:
        raise ValueError(
            f"{split_name} row count mismatch. "
            f"Expected {expected_rows:,}, "
            f"found {len(df):,}."
        )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = (
        ALL_FEATURES
        + [
            TARGET,
            "commit_id",
            "project",
        ]
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{split_name} is missing required columns:\n"
            f"{missing}"
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

    # --------------------------------------------------------
    # Normalize target
    # --------------------------------------------------------

    df[TARGET] = df[TARGET].apply(
        normalize_buggy
    )

    if df[TARGET].isna().any():
        missing_labels = int(
            df[TARGET].isna().sum()
        )

        raise ValueError(
            f"{split_name} contains "
            f"{missing_labels} missing buggy labels."
        )

    df[TARGET] = df[TARGET].astype(int)

    unique_targets = set(
        df[TARGET].unique()
    )

    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            f"Unexpected target values in "
            f"{split_name}: {unique_targets}"
        )

    # --------------------------------------------------------
    # Convert all model features to numeric
    # --------------------------------------------------------

    for feature in ALL_FEATURES:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Replace infinite values
    # --------------------------------------------------------

    df[ALL_FEATURES] = df[
        ALL_FEATURES
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # --------------------------------------------------------
    # Report missing values
    # --------------------------------------------------------

    missing_counts = df[
        ALL_FEATURES
    ].isna().sum()

    total_missing = int(
        missing_counts.sum()
    )

    if total_missing > 0:

        print(
            f"Missing model-feature values: "
            f"{total_missing:,}"
        )

        print(
            "Missing values will be filled "
            "using training-derived medians."
        )

    else:

        print(
            "Missing model-feature values: 0"
        )

    return df


def verify_no_cross_split_overlap(
    train_df,
    validation_df,
    test_df,
):
    """
    Verify that no commit appears in more than one split.
    """

    print("\n" + "=" * 70)
    print("COMMIT-ID OVERLAP VERIFICATION")
    print("=" * 70)

    train_ids = set(
        train_df["commit_id"]
    )

    validation_ids = set(
        validation_df["commit_id"]
    )

    test_ids = set(
        test_df["commit_id"]
    )

    train_val = train_ids & validation_ids
    train_test = train_ids & test_ids
    validation_test = (
        validation_ids & test_ids
    )

    print(
        f"Train / Validation: "
        f"{len(train_val)}"
    )

    print(
        f"Train / Test:       "
        f"{len(train_test)}"
    )

    print(
        f"Validation / Test:  "
        f"{len(validation_test)}"
    )

    if (
        train_val
        or train_test
        or validation_test
    ):
        raise ValueError(
            "Commit-ID overlap detected "
            "between chronological splits."
        )

    print(
        "\n[OK] No cross-split commit overlap."
    )


def calculate_metrics(
    y_true,
    y_pred,
    y_prob,
):
    """
    Calculate all experiment metrics.
    """

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "mcc": matthews_corrcoef(
            y_true,
            y_pred,
        ),

        "roc_auc": roc_auc_score(
            y_true,
            y_prob,
        ),

        "pr_auc": average_precision_score(
            y_true,
            y_prob,
        ),
    }

    return metrics


def evaluate_model(
    model,
    X,
    y,
    split_name,
):
    """
    Evaluate a trained model.
    """

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

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

    print("\n" + "=" * 70)
    print(
        f"{split_name.upper()} RESULTS"
    )
    print("=" * 70)

    for name, value in metrics.items():

        print(
            f"{name.upper():12s}: "
            f"{value:.4f}"
        )

    print("\nConfusion Matrix:")
    print(cm)

    return (
        metrics,
        predictions,
        probabilities,
        cm,
    )


def save_predictions(
    df,
    predictions,
    probabilities,
    model_name,
    split_name,
):
    """
    Save commit-level predictions.
    """

    columns = [
        "commit_id",
        "project",
        "author_date",
        TARGET,
    ]

    # author_date may not exist in some future
    # modified dataset, so handle gracefully.
    if "author_date" not in df.columns:

        columns.remove("author_date")

    output = df[
        columns
    ].copy()

    output["prediction"] = (
        predictions
    )

    output["probability"] = (
        probabilities
    )

    output_path = (
        PREDICTION_DIR
        / (
            f"{model_name}_"
            f"{split_name}_predictions.csv"
        )
    )

    output.to_csv(
        output_path,
        index=False,
    )

    return output_path


def save_confusion_matrix(
    cm,
    model_name,
    split_name,
):
    """
    Save confusion matrix as CSV.
    """

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
        CONFUSION_DIR
        / (
            f"{model_name}_"
            f"{split_name}_"
            f"confusion_matrix.csv"
        )
    )

    cm_df.to_csv(path)

    return path


def print_target_distribution(
    y_train,
    y_validation,
    y_test,
):
    """
    Print target distribution for all splits.
    """

    print("\n" + "=" * 70)
    print("TARGET DISTRIBUTION")
    print("=" * 70)

    for name, y in [
        ("Train", y_train),
        ("Validation", y_validation),
        ("Test", y_test),
    ]:

        buggy_count = int(
            np.sum(y == 1)
        )

        total = len(y)

        percentage = (
            100.0
            * buggy_count
            / total
        )

        print(
            f"{name:12s}: "
            f"buggy={buggy_count:,} / "
            f"{total:,} "
            f"({percentage:.2f}%)"
        )


def verify_feature_block():
    """
    Verify the expected feature dimensions.
    """

    print("\n" + "=" * 70)
    print("FEATURE CONFIGURATION")
    print("=" * 70)

    print(
        f"JIT features:  "
        f"{len(JIT_FEATURES)}"
    )

    print(
        f"LLM features:  "
        f"{len(LLM_FEATURES)}"
    )

    print(
        f"Total features:"
        f" {len(ALL_FEATURES)}"
    )

    if (
        len(JIT_FEATURES)
        != EXPECTED_JIT_FEATURE_COUNT
    ):
        raise ValueError(
            "Unexpected JIT feature count."
        )

    if (
        len(LLM_FEATURES)
        != EXPECTED_LLM_FEATURE_COUNT
    ):
        raise ValueError(
            "Unexpected LLM feature count."
        )

    if (
        len(ALL_FEATURES)
        != EXPECTED_TOTAL_FEATURE_COUNT
    ):
        raise ValueError(
            "Unexpected total feature count."
        )

    if len(
        set(ALL_FEATURES)
    ) != len(ALL_FEATURES):

        raise ValueError(
            "Duplicate feature names detected."
        )

    print(
        "[OK] 12 JIT + 51 LLM = 63 features."
    )


# ============================================================
# ROC / PR PLOTTING
# ============================================================

def plot_roc_pr(
    y_true,
    probabilities,
    model_name,
    split_name,
):
    """
    Save ROC and Precision-Recall curves for a trained model.
    PR-AUC label uses average precision, matching the reported metric.
    """
    safe_model_name = model_name.replace(" ", "_").lower()
    safe_split_name = split_name.lower()

    # --------------------------------------------------------
    # ROC curve
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Precision-Recall curve
    # --------------------------------------------------------
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


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print(
        "EXPERIMENT 5: JIT + LLM FEATURE FUSION"
    )
    print("=" * 70)

    print("\nMethodology:")
    print(
        "  Split: Project-wise chronological 70/15/15"
    )
    print(
        "  JIT features: 12"
    )
    print(
        "  LLM features: 51"
    )
    print(
        "  Total model features: 63"
    )
    print(
        "  Embeddings: NOT USED DIRECTLY"
    )
    print(
        "  PCA: NOT USED"
    )
    print(
        "  Resampling: NOT USED"
    )
    print(
        "  Random train/test split: NOT USED"
    )
    print(
        "  Test set: final evaluation only"
    )

    print("=" * 70)

    # ========================================================
    # 1. VERIFY FEATURE CONFIGURATION
    # ========================================================

    verify_feature_block()

    # ========================================================
    # 2. LOAD DATA
    # ========================================================

    train_df = load_split(
        TRAIN_PATH,
        "train",
    )

    validation_df = load_split(
        VALIDATION_PATH,
        "validation",
    )

    test_df = load_split(
        TEST_PATH,
        "test",
    )

    # ========================================================
    # 3. VERIFY SPLITS
    # ========================================================

    print("\n" + "=" * 70)
    print("SPLIT VERIFICATION")
    print("=" * 70)

    total_rows = (
        len(train_df)
        + len(validation_df)
        + len(test_df)
    )

    print(
        f"Train:       "
        f"{len(train_df):,}"
    )

    print(
        f"Validation:  "
        f"{len(validation_df):,}"
    )

    print(
        f"Test:        "
        f"{len(test_df):,}"
    )

    print(
        f"Total:       "
        f"{total_rows:,}"
    )

    if total_rows != 59996:

        raise ValueError(
            f"Expected 59,996 total rows, "
            f"found {total_rows:,}."
        )

    # ========================================================
    # 4. VERIFY NO ID OVERLAP
    # ========================================================

    verify_no_cross_split_overlap(
        train_df,
        validation_df,
        test_df,
    )

    # ========================================================
    # 5. PREPARE MODEL FEATURES
    # ========================================================

    print("\n" + "=" * 70)
    print("PREPARING MODEL FEATURES")
    print("=" * 70)

    # Explicitly select ONLY the 63 model features.
    #
    # This prevents metadata such as:
    # commit_id
    # project
    # message
    # diff
    # fix
    # author_date
    #
    # from entering the model.

    X_train = train_df[
        ALL_FEATURES
    ].copy()

    X_validation = validation_df[
        ALL_FEATURES
    ].copy()

    X_test = test_df[
        ALL_FEATURES
    ].copy()

    y_train = train_df[
        TARGET
    ].values

    y_validation = validation_df[
        TARGET
    ].values

    y_test = test_df[
        TARGET
    ].values

    print(
        f"X_train shape:       "
        f"{X_train.shape}"
    )

    print(
        f"X_validation shape:  "
        f"{X_validation.shape}"
    )

    print(
        f"X_test shape:        "
        f"{X_test.shape}"
    )

    # ========================================================
    # 6. TRAINING-ONLY MEDIAN IMPUTATION
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINING-ONLY MEDIAN IMPUTATION")
    print("=" * 70)

    print(
        "Calculating feature medians "
        "ONLY from training data."
    )

    train_medians = X_train.median()

    X_train = X_train.fillna(
        train_medians
    )

    X_validation = X_validation.fillna(
        train_medians
    )

    X_test = X_test.fillna(
        train_medians
    )

    # --------------------------------------------------------
    # Verify no NaN remains
    # --------------------------------------------------------

    if (
        X_train.isna().any().any()
        or X_validation.isna().any().any()
        or X_test.isna().any().any()
    ):

        remaining = {}

        for name, X in [
            ("train", X_train),
            ("validation", X_validation),
            ("test", X_test),
        ]:

            missing = X.isna().sum()

            missing = missing[
                missing > 0
            ]

            if len(missing) > 0:
                remaining[name] = (
                    missing.to_dict()
                )

        raise ValueError(
            "NaN values remain after "
            "training-derived median "
            f"imputation: {remaining}"
        )

    # --------------------------------------------------------
    # Verify finite values
    # --------------------------------------------------------

    for name, X in [
        ("train", X_train),
        ("validation", X_validation),
        ("test", X_test),
    ]:

        if not np.isfinite(
            X.values
        ).all():

            raise ValueError(
                f"Non-finite values remain "
                f"in {name} feature matrix."
            )

    print(
        "[OK] No NaN or infinite values "
        "remain in model features."
    )

    # ========================================================
    # 7. TARGET DISTRIBUTION
    # ========================================================

    print_target_distribution(
        y_train,
        y_validation,
        y_test,
    )

    # ========================================================
    # 8. CLASS IMBALANCE
    # ========================================================

    negative_count = np.sum(
        y_train == 0
    )

    positive_count = np.sum(
        y_train == 1
    )

    if positive_count == 0:

        raise ValueError(
            "Training set contains "
            "no positive buggy examples."
        )

    scale_pos_weight = (
        negative_count
        / positive_count
    )

    print("\n" + "=" * 70)
    print("CLASS IMBALANCE")
    print("=" * 70)

    print(
        f"Training negatives: "
        f"{negative_count:,}"
    )

    print(
        f"Training positives: "
        f"{positive_count:,}"
    )

    print(
        f"XGBoost scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    # ========================================================
    # 9. MODEL DEFINITIONS
    # ========================================================

    print("\n" + "=" * 70)
    print("MODEL CONFIGURATION")
    print("=" * 70)

    models = {}

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    models["random_forest"] = (
        RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    models["xgboost"] = (
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
        )
    )

    print(
        "Random Forest: configured"
    )

    print(
        "XGBoost: configured"
    )

    # ========================================================
    # 10. TRAIN + EVALUATE
    # ========================================================

    all_results = []

    for model_name, model in models.items():

        print("\n\n" + "#" * 70)
        print(
            f"TRAINING: {model_name.upper()}"
        )
        print("#" * 70)

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        print(
            "\nTraining model..."
        )

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

        validation_prediction_path = (
            save_predictions(
                validation_df,
                validation_predictions,
                validation_probabilities,
                model_name,
                "validation",
            )
        )

        validation_cm_path = (
            save_confusion_matrix(
                validation_cm,
                model_name,
                "validation",
            )
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

        test_prediction_path = (
            save_predictions(
                test_df,
                test_predictions,
                test_probabilities,
                model_name,
                "test",
            )
        )

        test_cm_path = (
            save_confusion_matrix(
                test_cm,
                model_name,
                "test",
            )
        )

        # ----------------------------------------------------
        # ROC / PR PLOTS
        # ----------------------------------------------------

        validation_roc_path, validation_pr_path = plot_roc_pr(
            y_validation,
            validation_probabilities,
            model_name,
            "validation",
        )

        test_roc_path, test_pr_path = plot_roc_pr(
            y_test,
            test_probabilities,
            model_name,
            "test",
        )

        print(
            f"\nValidation ROC plot: {validation_roc_path}"
        )
        print(
            f"Validation PR plot:  {validation_pr_path}"
        )
        print(
            f"Test ROC plot:       {test_roc_path}"
        )
        print(
            f"Test PR plot:        {test_pr_path}"
        )

        # ----------------------------------------------------
        # STORE RESULTS
        # ----------------------------------------------------

        result = {
            "model": model_name,

            "n_features": len(
                ALL_FEATURES
            ),

            "n_jit_features": len(
                JIT_FEATURES
            ),

            "n_llm_features": len(
                LLM_FEATURES
            ),

            # Validation
            "validation_accuracy":
                validation_metrics[
                    "accuracy"
                ],

            "validation_precision":
                validation_metrics[
                    "precision"
                ],

            "validation_recall":
                validation_metrics[
                    "recall"
                ],

            "validation_f1":
                validation_metrics[
                    "f1"
                ],

            "validation_mcc":
                validation_metrics[
                    "mcc"
                ],

            "validation_roc_auc":
                validation_metrics[
                    "roc_auc"
                ],

            "validation_pr_auc":
                validation_metrics[
                    "pr_auc"
                ],

            # Test
            "test_accuracy":
                test_metrics[
                    "accuracy"
                ],

            "test_precision":
                test_metrics[
                    "precision"
                ],

            "test_recall":
                test_metrics[
                    "recall"
                ],

            "test_f1":
                test_metrics[
                    "f1"
                ],

            "test_mcc":
                test_metrics[
                    "mcc"
                ],

            "test_roc_auc":
                test_metrics[
                    "roc_auc"
                ],

            "test_pr_auc":
                test_metrics[
                    "pr_auc"
                ],
        }

        all_results.append(result)

        # ----------------------------------------------------
        # SAVE MODEL
        # ----------------------------------------------------

        model_path = (
            MODEL_DIR
            / f"{model_name}.joblib"
        )

        joblib.dump(
            model,
            model_path,
        )

        print(
            f"\nModel saved:"
            f"\n  {model_path}"
        )

        print(
            f"\nValidation predictions:"
            f"\n  {validation_prediction_path}"
        )

        print(
            f"Validation confusion matrix:"
            f"\n  {validation_cm_path}"
        )

        print(
            f"\nTest predictions:"
            f"\n  {test_prediction_path}"
        )

        print(
            f"Test confusion matrix:"
            f"\n  {test_cm_path}"
        )

    # ========================================================
    # 11. SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    results_path = (
        METRICS_DIR
        / "experiment_5_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    # ========================================================
    # 12. SAVE CONFIGURATION
    # ========================================================

    configuration = {
        "experiment":
            "Experiment 5 - JIT + LLM",

        "description":
            "Traditional JIT metrics fused "
            "with LLM-derived semantic features.",

        "split_method":
            "Project-wise chronological 70/15/15",

        "train_rows":
            int(len(train_df)),

        "validation_rows":
            int(len(validation_df)),

        "test_rows":
            int(len(test_df)),

        "total_rows":
            int(total_rows),

        "jit_feature_count":
            len(JIT_FEATURES),

        "llm_feature_count":
            len(LLM_FEATURES),

        "total_feature_count":
            len(ALL_FEATURES),

        "jit_features":
            JIT_FEATURES,

        "llm_features":
            LLM_FEATURES,

        "target":
            TARGET,

        "threshold":
            0.5,

        "random_state":
            42,

        "embeddings_used_directly":
            False,

        "pca_used":
            False,

        "resampling_used":
            False,

        "random_split_used":
            False,

        "test_used_for_training":
            False,

        "median_imputation":
            True,

        "median_imputation_fit_on":
            "training data only",

        "xgboost_scale_pos_weight":
            float(scale_pos_weight),

        "models": [
            "Random Forest",
            "XGBoost",
        ],

        "feature_block": {
            "jit":
                JIT_FEATURES,
            "llm":
                LLM_FEATURES,
            "total":
                ALL_FEATURES,
        },

        "source_dataset":
            str(DATASET_DIR),
    }

    config_path = (
        OUTPUT_DIR
        / "experiment_5_configuration.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            configuration,
            f,
            indent=4,
        )

    # ========================================================
    # 13. SAVE FEATURE MEDIANS
    # ========================================================

    medians_path = (
        OUTPUT_DIR
        / "training_feature_medians.csv"
    )

    train_medians.to_csv(
        medians_path,
        header=["median"],
    )

    # ========================================================
    # 14. FINAL SUMMARY
    # ========================================================

    print("\n\n" + "=" * 70)
    print(
        "EXPERIMENT 5 COMPLETED"
    )
    print("=" * 70)

    print(
        "\nFeature configuration:"
    )

    print(
        f"  JIT features:   "
        f"{len(JIT_FEATURES)}"
    )

    print(
        f"  LLM features:   "
        f"{len(LLM_FEATURES)}"
    )

    print(
        f"  Total features: "
        f"{len(ALL_FEATURES)}"
    )

    print(
        "\nDataset:"
    )

    print(
        f"  Train:       "
        f"{len(train_df):,}"
    )

    print(
        f"  Validation:  "
        f"{len(validation_df):,}"
    )

    print(
        f"  Test:        "
        f"{len(test_df):,}"
    )

    print(
        f"  Total:       "
        f"{total_rows:,}"
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
        ].to_string(
            index=False
        )
    )

    print(
        "\nResults saved to:"
    )

    print(
        f"  {results_path}"
    )

    print(
        "\nConfiguration saved to:"
    )

    print(
        f"  {config_path}"
    )

    print(
        "\nModels saved to:"
    )

    print(
        f"  {MODEL_DIR}"
    )

    print(
        "\nPredictions saved to:"
    )

    print(
        f"  {PREDICTION_DIR}"
    )

    print(
        "\nConfusion matrices saved to:"
    )

    print(
        f"  {CONFUSION_DIR}"
    )

    print(
        "\nPlots saved to:"
    )

    print(
        f"  {PLOTS_DIR}"
    )

    print(
        "\n" + "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()