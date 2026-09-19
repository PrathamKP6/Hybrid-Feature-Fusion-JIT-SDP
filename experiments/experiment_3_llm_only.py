import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from xgboost import XGBClassifier

import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


# ============================================================
# EXPERIMENT 3: LLM ONLY
# ============================================================

# Purpose:
#   Evaluate software defect prediction using ONLY
#   structured LLM-derived semantic reasoning features.
#
# Methodology:
#   - Project-wise chronological 70/15/15 split
#   - 51 LLM features after one-hot encoding
#   - No JIT features
#   - No CodeBERT embeddings
#   - No PCA
#   - No resampling
#   - No random train/test split
#   - OneHotEncoder FIT ON TRAIN ONLY
#   - Median imputation FIT ON TRAIN ONLY
#   - Validation evaluated separately
#   - Test used only for final evaluation
#   - Fixed classification threshold = 0.5


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# IMPORTANT:
# This should point to the RAW LLM dataset containing:
# commit_id, categorical LLM fields, numerical LLM fields, buggy
DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "llm_experiment_dataset"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "experiment_3_llm_only"
)

MODEL_DIR = OUTPUT_DIR / "models"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
CONFUSION_DIR = OUTPUT_DIR / "confusion_matrices"
METRICS_DIR = OUTPUT_DIR / "metrics"

for directory in [
    OUTPUT_DIR,
    MODEL_DIR,
    PREDICTION_DIR,
    CONFUSION_DIR,
    METRICS_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEST_PATH = DATA_DIR / "test.csv"

RANDOM_STATE = 42
THRESHOLD = 0.5


# ============================================================
# EXPECTED SPLIT SIZES
# ============================================================

EXPECTED_ROWS = {
    "train": 41998,
    "validation": 8998,
    "test": 9000,
}


# ============================================================
# RAW LLM CATEGORICAL FEATURES
# ============================================================

LLM_CATEGORICAL_COLUMNS = [
    "intent",
    "change",
    "risk",
    "complexity",
    "scope",
    "test",
    "security",
]


# ============================================================
# LLM NUMERICAL FEATURES
# ============================================================

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


TARGET = "buggy"
COMMIT_ID = "commit_id"


# ============================================================
# EXPECTED ONE-HOT FEATURE COUNT
# ============================================================

EXPECTED_CATEGORICAL_FEATURE_COUNT = 37
EXPECTED_TOTAL_FEATURE_COUNT = 51


# ============================================================
# TARGET NORMALIZATION
# ============================================================

def normalize_buggy(value):

    if pd.isna(value):
        return np.nan

    if isinstance(value, (bool, np.bool_)):
        return int(value)

    if isinstance(
        value,
        (int, float, np.integer, np.floating),
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
# LOAD RAW SPLIT
# ============================================================

def load_split(path, split_name):

    print("\n" + "=" * 70)
    print(f"LOADING {split_name.upper()} DATA")
    print("=" * 70)

    print(f"Path: {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset file: {path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # --------------------------------------------------------
    # Row count
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
        LLM_CATEGORICAL_COLUMNS
        + LLM_NUMERICAL_FEATURES
        + [
            TARGET,
            COMMIT_ID,
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
        df[COMMIT_ID].duplicated().sum()
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
    # Convert numerical features
    # --------------------------------------------------------

    for feature in LLM_NUMERICAL_FEATURES:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Replace infinities
    # --------------------------------------------------------

    df[LLM_NUMERICAL_FEATURES] = (
        df[LLM_NUMERICAL_FEATURES]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    return df


# ============================================================
# CROSS-SPLIT COMMIT VERIFICATION
# ============================================================

def verify_no_cross_split_overlap(
    train_df,
    validation_df,
    test_df,
):

    print("\n" + "=" * 70)
    print("COMMIT-ID OVERLAP VERIFICATION")
    print("=" * 70)

    train_ids = set(
        train_df[COMMIT_ID]
    )

    validation_ids = set(
        validation_df[COMMIT_ID]
    )

    test_ids = set(
        test_df[COMMIT_ID]
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


# ============================================================
# ONE-HOT ENCODING
# ============================================================

def create_one_hot_encoder():

    print("\n" + "=" * 70)
    print("ONE-HOT ENCODER")
    print("=" * 70)

    print(
        "Encoder will be FIT ON TRAINING DATA ONLY."
    )

    # handle_unknown="ignore" is important:
    # if a category appears in validation/test that
    # was not present during training, it will not crash.
    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float64,
        )
    except TypeError:
        # Compatibility with older scikit-learn
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
            dtype=np.float64,
        )

    return encoder


def get_one_hot_feature_names(
    encoder
):

    feature_names = list(
        encoder.get_feature_names_out(
            LLM_CATEGORICAL_COLUMNS
        )
    )

    return feature_names


# ============================================================
# PREPARE MODEL FEATURES
# ============================================================

def prepare_features(
    train_df,
    validation_df,
    test_df,
):

    print("\n" + "=" * 70)
    print("PREPARING MODEL FEATURES")
    print("=" * 70)

    # --------------------------------------------------------
    # CATEGORICAL DATA
    # --------------------------------------------------------

    X_train_cat = train_df[
        LLM_CATEGORICAL_COLUMNS
    ].copy()

    X_validation_cat = validation_df[
        LLM_CATEGORICAL_COLUMNS
    ].copy()

    X_test_cat = test_df[
        LLM_CATEGORICAL_COLUMNS
    ].copy()

    # Fill missing categorical values.
    # "UNKNOWN" is treated as another category.
    X_train_cat = X_train_cat.fillna("UNKNOWN")
    X_validation_cat = X_validation_cat.fillna("UNKNOWN")
    X_test_cat = X_test_cat.fillna("UNKNOWN")

    # Convert everything to string to ensure
    # consistent categorical processing.
    for column in LLM_CATEGORICAL_COLUMNS:

        X_train_cat[column] = (
            X_train_cat[column].astype(str)
        )

        X_validation_cat[column] = (
            X_validation_cat[column].astype(str)
        )

        X_test_cat[column] = (
            X_test_cat[column].astype(str)
        )

    # --------------------------------------------------------
    # FIT ENCODER ONLY ON TRAIN
    # --------------------------------------------------------

    encoder = create_one_hot_encoder()

    X_train_cat_encoded = (
        encoder.fit_transform(
            X_train_cat
        )
    )

    X_validation_cat_encoded = (
        encoder.transform(
            X_validation_cat
        )
    )

    X_test_cat_encoded = (
        encoder.transform(
            X_test_cat
        )
    )

    categorical_feature_names = (
        get_one_hot_feature_names(
            encoder
        )
    )

    print(
        f"One-hot categorical features: "
        f"{len(categorical_feature_names)}"
    )

    if len(categorical_feature_names) != EXPECTED_CATEGORICAL_FEATURE_COUNT:

        raise ValueError(
            "Expected exactly 37 one-hot categorical "
            f"features, found "
            f"{len(categorical_feature_names)}.\n\n"
            f"Generated features:\n"
            f"{categorical_feature_names}"
        )

    print(
        "[OK] 37 categorical one-hot features."
    )

    # --------------------------------------------------------
    # NUMERICAL FEATURES
    # --------------------------------------------------------

    X_train_num = train_df[
        LLM_NUMERICAL_FEATURES
    ].copy()

    X_validation_num = validation_df[
        LLM_NUMERICAL_FEATURES
    ].copy()

    X_test_num = test_df[
        LLM_NUMERICAL_FEATURES
    ].copy()

    # --------------------------------------------------------
    # TRAIN-ONLY MEDIAN IMPUTATION
    # --------------------------------------------------------

    print(
        "\nCalculating numerical feature medians "
        "ONLY from training data."
    )

    train_medians = X_train_num.median()

    X_train_num = X_train_num.fillna(
        train_medians
    )

    X_validation_num = (
        X_validation_num.fillna(
            train_medians
        )
    )

    X_test_num = X_test_num.fillna(
        train_medians
    )

    # --------------------------------------------------------
    # CONVERT TO NUMPY
    # --------------------------------------------------------

    X_train_num = X_train_num.to_numpy(
        dtype=np.float64
    )

    X_validation_num = (
        X_validation_num.to_numpy(
            dtype=np.float64
        )
    )

    X_test_num = X_test_num.to_numpy(
        dtype=np.float64
    )

    # --------------------------------------------------------
    # COMBINE CATEGORICAL + NUMERICAL
    # --------------------------------------------------------

    X_train = np.hstack([
        X_train_cat_encoded,
        X_train_num,
    ])

    X_validation = np.hstack([
        X_validation_cat_encoded,
        X_validation_num,
    ])

    X_test = np.hstack([
        X_test_cat_encoded,
        X_test_num,
    ])

    # --------------------------------------------------------
    # FINAL FEATURE NAMES
    # --------------------------------------------------------

    feature_names = (
        categorical_feature_names
        + LLM_NUMERICAL_FEATURES
    )

    if len(feature_names) != EXPECTED_TOTAL_FEATURE_COUNT:

        raise ValueError(
            "Expected exactly 51 final LLM features, "
            f"found {len(feature_names)}."
        )

    print(
        f"\nFinal LLM feature count: "
        f"{len(feature_names)}"
    )

    print(
        "[OK] 37 categorical + "
        "14 numerical = 51 features."
    )

    # --------------------------------------------------------
    # VERIFY FINITE
    # --------------------------------------------------------

    for name, X in [
        ("train", X_train),
        ("validation", X_validation),
        ("test", X_test),
    ]:

        if not np.isfinite(X).all():

            raise ValueError(
                f"Non-finite values remain "
                f"in {name} feature matrix."
            )

    print(
        "[OK] No NaN or infinite values "
        "remain in model features."
    )

    return (
        X_train,
        X_validation,
        X_test,
        feature_names,
        encoder,
        train_medians,
    )


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

def print_target_distribution(
    y_train,
    y_validation,
    y_test,
):

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


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_probability,
):

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
            y_probability,
        ),

        "pr_auc": average_precision_score(
            y_true,
            y_probability,
        ),
    }

    return metrics


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    split_name,
):

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    metrics = calculate_metrics(
        y,
        predictions,
        probabilities,
    )

    cm = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
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

    columns = [
        COMMIT_ID,
        TARGET,
    ]

    if "project" in df.columns:
        columns.insert(
            1,
            "project",
        )

    if "author_date" in df.columns:
        columns.insert(
            2 if "project" in df.columns else 1,
            "author_date",
        )

    output = df[
        columns
    ].copy()

    output["prediction"] = predictions

    output["probability"] = probabilities

    output_dir = (
        PREDICTION_DIR
        / model_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{split_name}_predictions.csv"
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

    output_dir = (
        CONFUSION_DIR
        / model_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        output_dir
        / f"{split_name}_confusion_matrix.csv"
    )

    cm_df.to_csv(path)

    return path


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(
    model,
    feature_names,
    model_name,
):

    importance = pd.DataFrame({

        "feature": feature_names,

        "importance":
            model.feature_importances_,

    })

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False,
        )
    )

    path = (
        OUTPUT_DIR
        / f"{model_name}_feature_importance.csv"
    )

    importance.to_csv(
        path,
        index=False,
    )

    # Top 20 plot

    top_features = (
        importance
        .head(20)
        .sort_values("importance")
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        top_features["feature"],
        top_features["importance"],
    )

    plt.xlabel(
        "Feature Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"{model_name.replace('_', ' ').title()} "
        "- Top 20 Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / f"{model_name}_feature_importance.png",
        dpi=300,
    )

    plt.close()


# ============================================================
# SAVE ROC / PR CURVES
# ============================================================

def save_curves(
    y_true,
    probability,
    model_name,
):

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    fpr, tpr, _ = roc_curve(
        y_true,
        probability,
    )

    roc_auc = roc_auc_score(
        y_true,
        probability,
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {roc_auc:.4f}",
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
        f"{model_name} - ROC Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / f"{model_name.lower().replace(' ', '_')}_roc.png",
        dpi=300,
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

    pr_auc = average_precision_score(
        y_true,
        probability,
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        recall,
        precision,
        label=f"PR-AUC = {pr_auc:.4f}",
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        f"{model_name} - Precision-Recall Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / f"{model_name.lower().replace(' ', '_')}_pr.png",
        dpi=300,
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 70)
    print(
        "EXPERIMENT 3: LLM-ONLY FEATURE PREDICTION"
    )
    print("=" * 70)

    print("\nMethodology:")

    print(
        "  Split: Project-wise chronological 70/15/15"
    )

    print(
        "  LLM features: 51"
    )

    print(
        "  JIT features: 0"
    )

    print(
        "  CodeBERT embeddings: NOT USED"
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
        "  One-hot encoding: TRAIN ONLY"
    )

    print(
        "  Median imputation: TRAIN ONLY"
    )

    print(
        "  Test set: final evaluation only"
    )

    print(
        "  Threshold: 0.5"
    )

    print("=" * 70)


    # ========================================================
    # 1. LOAD DATA
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
    # 2. VERIFY TOTAL ROW COUNT
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
    # 3. VERIFY CROSS-SPLIT OVERLAP
    # ========================================================

    verify_no_cross_split_overlap(
        train_df,
        validation_df,
        test_df,
    )


    # ========================================================
    # 4. PREPARE MODEL FEATURES
    # ========================================================

    (
        X_train,
        X_validation,
        X_test,
        feature_names,
        encoder,
        train_medians,
    ) = prepare_features(
        train_df,
        validation_df,
        test_df,
    )

    print(
        f"\nX_train shape:       "
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
    # 5. TARGET
    # ========================================================

    y_train = train_df[
        TARGET
    ].values

    y_validation = validation_df[
        TARGET
    ].values

    y_test = test_df[
        TARGET
    ].values

    print_target_distribution(
        y_train,
        y_validation,
        y_test,
    )


    # ========================================================
    # 6. CLASS IMBALANCE
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
    # 7. MODEL CONFIGURATION
    # ========================================================

    print("\n" + "=" * 70)
    print("MODEL CONFIGURATION")
    print("=" * 70)

    models = {

        "random_forest":
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                class_weight="balanced",
                random_state=RANDOM_STATE,
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
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist",
            ),
    }

    print(
        "Random Forest: configured"
    )

    print(
        "XGBoost: configured"
    )


    # ========================================================
    # 8. TRAIN + VALIDATE + TEST
    # ========================================================

    all_results = []

    for model_name, model in models.items():

        print("\n\n" + "#" * 70)

        print(
            f"TRAINING: "
            f"{model_name.upper()}"
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
        # STORE RESULTS
        # ----------------------------------------------------

        result = {

            "experiment":
                "experiment_3_llm_only",

            "model":
                model_name,

            "n_features":
                len(feature_names),

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

        # ----------------------------------------------------
        # FEATURE IMPORTANCE
        # ----------------------------------------------------

        save_feature_importance(
            model,
            feature_names,
            model_name,
        )

        # ----------------------------------------------------
        # ROC / PR CURVES
        # ----------------------------------------------------

        save_curves(
            y_test,
            test_probabilities,
            model_name.replace(
                "_",
                " ",
            ).title(),
        )


    # ========================================================
    # 9. SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    results_path = (
        METRICS_DIR
        / "experiment_3_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )


    # ========================================================
    # 10. SAVE ENCODER
    # ========================================================

    encoder_path = (
        OUTPUT_DIR
        / "one_hot_encoder.joblib"
    )

    joblib.dump(
        encoder,
        encoder_path,
    )


    # ========================================================
    # 11. SAVE FEATURE NAMES
    # ========================================================

    feature_names_path = (
        OUTPUT_DIR
        / "llm_feature_names.json"
    )

    with open(
        feature_names_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            feature_names,
            f,
            indent=4,
        )


    # ========================================================
    # 12. SAVE TRAINING MEDIANS
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
    # 13. SAVE CONFIGURATION
    # ========================================================

    configuration = {

        "experiment":
            "Experiment 3 - LLM Only",

        "description":
            "LLM-derived structured semantic "
            "reasoning features without JIT "
            "or CodeBERT features.",

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

        "categorical_columns":
            LLM_CATEGORICAL_COLUMNS,

        "categorical_one_hot_feature_count":
            len(
                [
                    x
                    for x in feature_names
                    if x not in LLM_NUMERICAL_FEATURES
                ]
            ),

        "numerical_feature_count":
            len(LLM_NUMERICAL_FEATURES),

        "total_feature_count":
            len(feature_names),

        "one_hot_encoding":
            True,

        "one_hot_encoder_fit_on":
            "training data only",

        "unknown_categories":
            "ignore",

        "jit_features_used":
            False,

        "codebert_embeddings_used":
            False,

        "pca_used":
            False,

        "resampling_used":
            False,

        "random_split_used":
            False,

        "test_used_for_training":
            False,

        "target":
            TARGET,

        "threshold":
            THRESHOLD,

        "random_state":
            RANDOM_STATE,

        "median_imputation":
            True,

        "median_imputation_fit_on":
            "training data only",

        "xgboost_scale_pos_weight":
            float(scale_pos_weight),

        "models": {

            "random_forest": {
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "class_weight": "balanced",
                "random_state": RANDOM_STATE,
            },

            "xgboost": {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "eval_metric": "logloss",
                "scale_pos_weight":
                    float(scale_pos_weight),
                "random_state": RANDOM_STATE,
            },
        },

        "source_dataset":
            str(DATA_DIR),
    }

    config_path = (
        OUTPUT_DIR
        / "experiment_3_configuration.json"
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
    # 14. FINAL SUMMARY
    # ========================================================

    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "EXPERIMENT 3 COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        "\nFeature configuration:"
    )

    print(
        f"  Categorical columns: "
        f"{len(LLM_CATEGORICAL_COLUMNS)}"
    )

    print(
        f"  One-hot features: "
        f"{EXPECTED_CATEGORICAL_FEATURE_COUNT}"
    )

    print(
        f"  Numerical features: "
        f"{len(LLM_NUMERICAL_FEATURES)}"
    )

    print(
        f"  Total LLM features: "
        f"{len(feature_names)}"
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
        "\nOne-hot encoder saved to:"
    )

    print(
        f"  {encoder_path}"
    )

    print(
        "\nFeature names saved to:"
    )

    print(
        f"  {feature_names_path}"
    )

    print(
        "\nConfiguration saved to:"
    )

    print(
        f"  {config_path}"
    )

    print(
        "\n" + "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()