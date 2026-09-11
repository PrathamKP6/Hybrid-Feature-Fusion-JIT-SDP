"""
Experiment 4D — Stacking

JIT + CodeBERT Probability Stacking
------------------------------------

Base models:
    1. JIT features (12) -> Random Forest
    2. CodeBERT PCA features (384) -> Random Forest

    3. JIT features (12) -> XGBoost
    4. CodeBERT PCA features (384) -> XGBoost

Meta model:
    Logistic Regression

Stacking strategy:
    Base-model probabilities on VALIDATION
            |
            v
    Logistic Regression meta-model
            |
            v
    Test base-model probabilities
            |
            v
    Final test prediction

Important:
- Project-wise chronological 70/15/15 split already exists.
- CodeBERT uses the FULL 384-D PCA representation.
- No additional PCA.
- No feature selection.
- No resampling.
- No random splitting.
- Meta-model is trained on VALIDATION probabilities.
- TEST is used only for final evaluation.
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\Hybrid-Feature-Fusion-JIT-SDP"

JIT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "chronological_splits"
)

PCA_DIR = os.path.join(
    BASE_DIR,
    "results",
    "pca"
)

TRAIN_JIT_PATH = os.path.join(
    JIT_DIR,
    "train.csv"
)

VAL_JIT_PATH = os.path.join(
    JIT_DIR,
    "validation.csv"
)

TEST_JIT_PATH = os.path.join(
    JIT_DIR,
    "test.csv"
)

TRAIN_PCA_PATH = os.path.join(
    PCA_DIR,
    "train_pca384.csv"
)

VAL_PCA_PATH = os.path.join(
    PCA_DIR,
    "validation_pca384.csv"
)

TEST_PCA_PATH = os.path.join(
    PCA_DIR,
    "test_pca384.csv"
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "experiment_4d_stacking"
)

METRICS_DIR = os.path.join(
    OUTPUT_DIR,
    "metrics"
)

MODELS_DIR = os.path.join(
    OUTPUT_DIR,
    "models"
)

PREDICTIONS_DIR = os.path.join(
    OUTPUT_DIR,
    "predictions"
)

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)


# ============================================================
# PARAMETERS
# ============================================================

RANDOM_STATE = 42

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

CODEBERT_DIM = 384


# ============================================================
# TARGET NORMALIZATION
# ============================================================

def normalize_target(series):

    def convert(value):

        if pd.isna(value):
            return np.nan

        if isinstance(value, bool):
            return int(value)

        if isinstance(value, (int, np.integer)):
            return int(value)

        if isinstance(value, (float, np.floating)):
            return int(value)

        value = str(value).strip().lower()

        if value in {
            "1",
            "true",
            "yes",
            "buggy"
        }:
            return 1

        if value in {
            "0",
            "false",
            "no",
            "clean",
            "non-buggy"
        }:
            return 0

        raise ValueError(
            f"Unknown target value: {value}"
        )

    return series.apply(convert).astype(int)


# ============================================================
# EMBEDDING PARSER
# ============================================================

def parse_embedding(value):

    if isinstance(value, list):
        return np.asarray(
            value,
            dtype=np.float32
        )

    if isinstance(value, np.ndarray):
        return value.astype(
            np.float32
        )

    if pd.isna(value):
        return None

    try:

        parsed = json.loads(value)

        if isinstance(parsed, list):
            return np.asarray(
                parsed,
                dtype=np.float32
            )

    except Exception:
        pass

    return None


# ============================================================
# EXTRACT CODEBERT FEATURES
# ============================================================

def extract_codebert(df):

    vectors = []

    for value in df["embedding"]:

        vector = parse_embedding(value)

        if vector is None:
            raise ValueError(
                "Invalid CodeBERT embedding found."
            )

        if len(vector) != CODEBERT_DIM:
            raise ValueError(
                f"Expected {CODEBERT_DIM}-D embedding, "
                f"found {len(vector)} dimensions."
            )

        vectors.append(vector)

    X = np.vstack(vectors)

    if not np.isfinite(X).all():
        raise ValueError(
            "Non-finite values found in CodeBERT embeddings."
        )

    return X


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 80)
print("EXPERIMENT 4D — STACKING")
print("=" * 80)

print("\nLoading JIT datasets...")

train_jit = pd.read_csv(
    TRAIN_JIT_PATH
)

val_jit = pd.read_csv(
    VAL_JIT_PATH
)

test_jit = pd.read_csv(
    TEST_JIT_PATH
)

print(
    f"Train JIT: {len(train_jit):,}"
)

print(
    f"Val JIT  : {len(val_jit):,}"
)

print(
    f"Test JIT : {len(test_jit):,}"
)


print("\nLoading CodeBERT PCA datasets...")

train_pca = pd.read_csv(
    TRAIN_PCA_PATH
)

val_pca = pd.read_csv(
    VAL_PCA_PATH
)

test_pca = pd.read_csv(
    TEST_PCA_PATH
)

print(
    f"Train CodeBERT: {len(train_pca):,}"
)

print(
    f"Val CodeBERT  : {len(val_pca):,}"
)

print(
    f"Test CodeBERT : {len(test_pca):,}"
)


# ============================================================
# VALIDATE COMMIT IDS
# ============================================================

print("\nValidating commit IDs...")

for df in [
    train_jit,
    val_jit,
    test_jit,
    train_pca,
    val_pca,
    test_pca,
]:

    assert df["commit_id"].is_unique


assert set(
    train_jit["commit_id"]
) == set(
    train_pca["commit_id"]
)

assert set(
    val_jit["commit_id"]
) == set(
    val_pca["commit_id"]
)

assert set(
    test_jit["commit_id"]
) == set(
    test_pca["commit_id"]
)

print("Commit ID validation passed.")


# ============================================================
# VALIDATE LABEL CONSISTENCY
# ============================================================

def validate_labels(jit_df, pca_df, split_name):

    jit_labels = normalize_target(
        jit_df["buggy"]
    )

    pca_labels = normalize_target(
        pca_df["buggy"]
    )

    jit_map = dict(
        zip(
            jit_df["commit_id"],
            jit_labels
        )
    )

    pca_map = dict(
        zip(
            pca_df["commit_id"],
            pca_labels
        )
    )

    for commit_id in jit_map:

        assert (
            jit_map[commit_id]
            ==
            pca_map[commit_id]
        ), (
            f"{split_name}: label mismatch "
            f"for {commit_id}"
        )

    print(
        f"{split_name}: label validation passed."
    )


validate_labels(
    train_jit,
    train_pca,
    "TRAIN"
)

validate_labels(
    val_jit,
    val_pca,
    "VALIDATION"
)

validate_labels(
    test_jit,
    test_pca,
    "TEST"
)


# ============================================================
# PREPARE JIT FEATURES
# ============================================================

def prepare_jit(df):

    X = df[
        JIT_FEATURES
    ].copy()

    for feature in JIT_FEATURES:

        X[feature] = pd.to_numeric(
            X[feature],
            errors="coerce"
        )

    return X


X_train_jit = prepare_jit(
    train_jit
)

X_val_jit = prepare_jit(
    val_jit
)

X_test_jit = prepare_jit(
    test_jit
)


# ============================================================
# TRAIN-ONLY JIT MEDIAN IMPUTATION
# ============================================================

print(
    "\nApplying train-only JIT median imputation..."
)

jit_medians = X_train_jit.median()

X_train_jit = X_train_jit.fillna(
    jit_medians
)

X_val_jit = X_val_jit.fillna(
    jit_medians
)

X_test_jit = X_test_jit.fillna(
    jit_medians
)


X_train_jit = X_train_jit.values
X_val_jit = X_val_jit.values
X_test_jit = X_test_jit.values


# ============================================================
# PREPARE CODEBERT
# ============================================================

print(
    "\nExtracting full 384-D CodeBERT embeddings..."
)

X_train_codebert = extract_codebert(
    train_pca
)

X_val_codebert = extract_codebert(
    val_pca
)

X_test_codebert = extract_codebert(
    test_pca
)

print(
    "Train:",
    X_train_codebert.shape
)

print(
    "Validation:",
    X_val_codebert.shape
)

print(
    "Test:",
    X_test_codebert.shape
)

assert X_train_codebert.shape[1] == 384
assert X_val_codebert.shape[1] == 384
assert X_test_codebert.shape[1] == 384


# ============================================================
# TARGET
# ============================================================

y_train = normalize_target(
    train_jit["buggy"]
)

y_val = normalize_target(
    val_jit["buggy"]
)

y_test = normalize_target(
    test_jit["buggy"]
)


# ============================================================
# MODEL PARAMETERS
# ============================================================

negative_count = np.sum(
    y_train == 0
)

positive_count = np.sum(
    y_train == 1
)

scale_pos_weight = (
    negative_count /
    positive_count
)


# ============================================================
# FUNCTION TO TRAIN BASE MODELS
# ============================================================

def train_base_models(
    X_train,
    y_train,
    feature_name
):

    print("\n" + "=" * 80)

    print(
        f"TRAINING BASE MODELS — {feature_name}"
    )

    print("=" * 80)


    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    print(
        "\nTraining Random Forest..."
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    rf.fit(
        X_train,
        y_train
    )

    print(
        "Random Forest completed."
    )


    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    print(
        "\nTraining XGBoost..."
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
        n_jobs=-1
    )

    xgb.fit(
        X_train,
        y_train
    )

    print(
        "XGBoost completed."
    )

    return rf, xgb


# ============================================================
# TRAIN JIT BASE MODELS
# ============================================================

jit_rf, jit_xgb = train_base_models(
    X_train_jit,
    y_train,
    "JIT — 12 features"
)


# ============================================================
# TRAIN CODEBERT BASE MODELS
# ============================================================

codebert_rf, codebert_xgb = train_base_models(
    X_train_codebert,
    y_train,
    "CodeBERT — 384 features"
)


# ============================================================
# GENERATE VALIDATION BASE PROBABILITIES
# ============================================================

print("\n" + "=" * 80)
print("GENERATING VALIDATION BASE-MODEL PROBABILITIES")
print("=" * 80)

val_jit_rf_prob = jit_rf.predict_proba(
    X_val_jit
)[:, 1]

val_jit_xgb_prob = jit_xgb.predict_proba(
    X_val_jit
)[:, 1]

val_codebert_rf_prob = codebert_rf.predict_proba(
    X_val_codebert
)[:, 1]

val_codebert_xgb_prob = codebert_xgb.predict_proba(
    X_val_codebert
)[:, 1]


# ============================================================
# META FEATURES
# ============================================================

X_meta_val = np.column_stack([
    val_jit_rf_prob,
    val_codebert_rf_prob,
    val_jit_xgb_prob,
    val_codebert_xgb_prob,
])


meta_feature_names = [
    "jit_rf_probability",
    "codebert_rf_probability",
    "jit_xgb_probability",
    "codebert_xgb_probability",
]


print(
    "\nMeta-feature matrix:"
)

print(
    X_meta_val.shape
)

assert X_meta_val.shape == (
    len(y_val),
    4
)


# ============================================================
# TRAIN LOGISTIC REGRESSION META MODEL
# ============================================================

print("\n" + "=" * 80)
print("TRAINING LOGISTIC REGRESSION META MODEL")
print("=" * 80)

meta_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "logistic_regression",
        LogisticRegression(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=1000
        )
    )
])

meta_model.fit(
    X_meta_val,
    y_val
)

print(
    "Meta-model training completed."
)


# ============================================================
# GENERATE TEST BASE PROBABILITIES
# ============================================================

print("\n" + "=" * 80)
print("GENERATING TEST BASE-MODEL PROBABILITIES")
print("=" * 80)

test_jit_rf_prob = jit_rf.predict_proba(
    X_test_jit
)[:, 1]

test_codebert_rf_prob = codebert_rf.predict_proba(
    X_test_codebert
)[:, 1]

test_jit_xgb_prob = jit_xgb.predict_proba(
    X_test_jit
)[:, 1]

test_codebert_xgb_prob = codebert_xgb.predict_proba(
    X_test_codebert
)[:, 1]


X_meta_test = np.column_stack([
    test_jit_rf_prob,
    test_codebert_rf_prob,
    test_jit_xgb_prob,
    test_codebert_xgb_prob,
])


print(
    "\nTest meta-feature matrix:"
)

print(
    X_meta_test.shape
)

assert X_meta_test.shape == (
    len(y_test),
    4
)


# ============================================================
# FINAL STACKED PREDICTION
# ============================================================

test_stacking_probability = (
    meta_model.predict_proba(
        X_meta_test
    )[:, 1]
)

test_stacking_prediction = (
    test_stacking_probability >= 0.5
).astype(int)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_predictions(
    y_true,
    predictions,
    probabilities,
    model_name
):

    cm = confusion_matrix(
        y_true,
        predictions
    )

    results = {
        "experiment": "4D",
        "model": model_name,

        "accuracy": accuracy_score(
            y_true,
            predictions
        ),

        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "mcc": matthews_corrcoef(
            y_true,
            predictions
        ),

        "roc_auc": roc_auc_score(
            y_true,
            probabilities
        ),

        "pr_auc": average_precision_score(
            y_true,
            probabilities
        ),

        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    print(
        "\n" + "=" * 70
    )

    print(
        f"{model_name} — TEST"
    )

    print(
        "=" * 70
    )

    print(
        f"Accuracy : {results['accuracy']:.4f}"
    )

    print(
        f"Precision: {results['precision']:.4f}"
    )

    print(
        f"Recall   : {results['recall']:.4f}"
    )

    print(
        f"F1       : {results['f1']:.4f}"
    )

    print(
        f"MCC      : {results['mcc']:.4f}"
    )

    print(
        f"ROC-AUC  : {results['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC   : {results['pr_auc']:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(cm)

    return results


# ============================================================
# EVALUATE STACKING MODEL
# ============================================================

stacking_results = evaluate_predictions(
    y_test,
    test_stacking_prediction,
    test_stacking_probability,
    "Stacking — Logistic Regression"
)


# ============================================================
# ALSO SAVE BASE MODEL TEST PERFORMANCE
# ============================================================

base_results = []

base_results.append(
    evaluate_predictions(
        y_test,
        (
            test_jit_rf_prob >= 0.5
        ).astype(int),
        test_jit_rf_prob,
        "JIT — Random Forest"
    )
)

base_results.append(
    evaluate_predictions(
        y_test,
        (
            test_codebert_rf_prob >= 0.5
        ).astype(int),
        test_codebert_rf_prob,
        "CodeBERT — Random Forest"
    )
)

base_results.append(
    evaluate_predictions(
        y_test,
        (
            test_jit_xgb_prob >= 0.5
        ).astype(int),
        test_jit_xgb_prob,
        "JIT — XGBoost"
    )
)

base_results.append(
    evaluate_predictions(
        y_test,
        (
            test_codebert_xgb_prob >= 0.5
        ).astype(int),
        test_codebert_xgb_prob,
        "CodeBERT — XGBoost"
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

all_results = base_results + [
    stacking_results
]

results_df = pd.DataFrame(
    all_results
)

results_path = os.path.join(
    METRICS_DIR,
    "experiment_4d_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame({

    "commit_id":
        test_jit["commit_id"],

    "project":
        test_jit["project"],

    "actual_buggy":
        y_test,

    "jit_rf_probability":
        test_jit_rf_prob,

    "codebert_rf_probability":
        test_codebert_rf_prob,

    "jit_xgb_probability":
        test_jit_xgb_prob,

    "codebert_xgb_probability":
        test_codebert_xgb_prob,

    "stacking_probability":
        test_stacking_probability,

    "stacking_prediction":
        test_stacking_prediction,
})

predictions_path = os.path.join(
    PREDICTIONS_DIR,
    "test_stacking_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# SAVE MODELS
# ============================================================

joblib.dump(
    jit_rf,
    os.path.join(
        MODELS_DIR,
        "jit_random_forest.joblib"
    )
)

joblib.dump(
    jit_xgb,
    os.path.join(
        MODELS_DIR,
        "jit_xgboost.joblib"
    )
)

joblib.dump(
    codebert_rf,
    os.path.join(
        MODELS_DIR,
        "codebert_random_forest.joblib"
    )
)

joblib.dump(
    codebert_xgb,
    os.path.join(
        MODELS_DIR,
        "codebert_xgboost.joblib"
    )
)

joblib.dump(
    meta_model,
    os.path.join(
        MODELS_DIR,
        "logistic_regression_meta_model.joblib"
    )
)


# ============================================================
# SAVE META FEATURE INFORMATION
# ============================================================

meta_config = {

    "experiment": "4D",

    "method": "Probability Stacking",

    "base_models": [
        "JIT Random Forest",
        "CodeBERT Random Forest",
        "JIT XGBoost",
        "CodeBERT XGBoost",
    ],

    "meta_model":
        "Logistic Regression",

    "jit_features":
        JIT_FEATURES,

    "jit_dimension":
        12,

    "codebert_dimension":
        384,

    "codebert_feature_selection":
        False,

    "additional_transformation":
        False,

    "additional_pca":
        False,

    "resampling":
        False,

    "meta_training":
        "validation_probabilities",

    "test_usage":
        "final_evaluation_only",

    "threshold":
        0.5,

    "random_state":
        RANDOM_STATE,

    "split_strategy":
        "project-wise chronological 70/15/15",
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "experiment_4d_config.json"
    ),
    "w"
) as f:

    json.dump(
        meta_config,
        f,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("EXPERIMENT 4D COMPLETED")
print("=" * 80)

print(
    "\nArchitecture:"
)

print(
    "JIT (12) + CodeBERT (384)"
)

print(
    "        ↓"
)

print(
    "RF/XGBoost base probabilities"
)

print(
    "        ↓"
)

print(
    "Logistic Regression meta-model"
)

print(
    "\nFinal TEST results:"
)

print(
    results_df[
        [
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "mcc",
            "roc_auc",
            "pr_auc"
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nResults saved to:"
)

print(
    results_path
)

print(
    "\nPredictions saved to:"
)

print(
    predictions_path
)

print(
    "\nModels saved to:"
)

print(
    MODELS_DIR
)

print(
    "\n" + "=" * 80
)

print(
    "ALL VERIFICATIONS PASSED"
)

print(
    "=" * 80
)