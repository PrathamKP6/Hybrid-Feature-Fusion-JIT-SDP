"""
Experiment 4E — Probability Blending

Weighted JIT + CodeBERT Probability Blend
-----------------------------------------

Base models:
    JIT (12 features)
        -> Random Forest
        -> XGBoost

    CodeBERT (384 PCA features)
        -> Random Forest
        -> XGBoost

Blending:
    P_blend = alpha * P_JIT + (1-alpha) * P_CodeBERT

Weight selection:
    - Alpha is selected using VALIDATION F1 only.
    - TEST is never used to select the weight.
    - Once selected, alpha is frozen and applied to TEST.

Important:
- Project-wise chronological 70/15/15 split already exists.
- Full 384-D CodeBERT PCA representation is used.
- No additional PCA.
- No feature selection.
- No resampling.
- Same RF/XGBoost configurations as previous experiments.
- Threshold = 0.5.
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import joblib

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
    "experiment_4e_probability_blend"
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

BLEND_DIR = os.path.join(
    OUTPUT_DIR,
    "blend_analysis"
)

for directory in [
    OUTPUT_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    BLEND_DIR,
]:
    os.makedirs(
        directory,
        exist_ok=True
    )


# ============================================================
# PARAMETERS
# ============================================================

RANDOM_STATE = 42

CODEBERT_DIM = 384

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

THRESHOLD = 0.5

# Candidate weights for JIT.
# alpha = 1.0 -> 100% JIT
# alpha = 0.0 -> 100% CodeBERT
BLEND_WEIGHTS = np.arange(
    0.0,
    1.0001,
    0.05
)


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
# EXTRACT CODEBERT
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
            "Non-finite CodeBERT values found."
        )

    return X


# ============================================================
# PREPARE JIT
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


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 80)
print("EXPERIMENT 4E — PROBABILITY BLENDING")
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


print("\nLoading CodeBERT datasets...")

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
# VALIDATE IDS
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

print(
    "Commit ID validation passed."
)


# ============================================================
# VALIDATE LABELS
# ============================================================

def validate_labels(
    jit_df,
    pca_df,
    split_name
):

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
            f"{split_name}: "
            f"label mismatch for "
            f"{commit_id}"
        )

    print(
        f"{split_name}: "
        f"label validation passed."
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

print(
    "\nPreparing JIT features..."
)

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
# TRAIN-ONLY MEDIAN IMPUTATION
# ============================================================

print(
    "Applying train-only JIT median imputation..."
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
    "\nExtracting FULL 384-D CodeBERT representation..."
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
# CLASS WEIGHT FOR XGBOOST
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

print(
    f"\nXGBoost scale_pos_weight: "
    f"{scale_pos_weight:.4f}"
)


# ============================================================
# TRAIN BASE MODELS
# ============================================================

def train_models(
    X_train,
    y_train,
    branch_name
):

    print(
        "\n" + "=" * 80
    )

    print(
        f"TRAINING {branch_name} BASE MODELS"
    )

    print(
        "=" * 80
    )


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


# JIT models
jit_rf, jit_xgb = train_models(
    X_train_jit,
    y_train,
    "JIT — 12 features"
)


# CodeBERT models
codebert_rf, codebert_xgb = train_models(
    X_train_codebert,
    y_train,
    "CodeBERT — 384 features"
)


# ============================================================
# VALIDATION PROBABILITIES
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "GENERATING VALIDATION PROBABILITIES"
)

print(
    "=" * 80
)


val_jit_rf_prob = jit_rf.predict_proba(
    X_val_jit
)[:, 1]

val_codebert_rf_prob = codebert_rf.predict_proba(
    X_val_codebert
)[:, 1]

val_jit_xgb_prob = jit_xgb.predict_proba(
    X_val_jit
)[:, 1]

val_codebert_xgb_prob = codebert_xgb.predict_proba(
    X_val_codebert
)[:, 1]


# ============================================================
# FIND BEST BLEND WEIGHT
# ============================================================

def search_best_weight(
    jit_probability,
    codebert_probability,
    y_true,
    model_name
):

    rows = []

    print(
        "\n" + "=" * 80
    )

    print(
        f"SEARCHING VALIDATION WEIGHT — {model_name}"
    )

    print(
        "=" * 80
    )

    for alpha in BLEND_WEIGHTS:

        blended_probability = (
            alpha * jit_probability
            +
            (1.0 - alpha)
            * codebert_probability
        )

        predictions = (
            blended_probability >= THRESHOLD
        ).astype(int)

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        mcc = matthews_corrcoef(
            y_true,
            predictions
        )

        roc_auc = roc_auc_score(
            y_true,
            blended_probability
        )

        pr_auc = average_precision_score(
            y_true,
            blended_probability
        )

        rows.append({
            "alpha_jit": alpha,
            "weight_jit": alpha,
            "weight_codebert": 1.0 - alpha,
            "f1": f1,
            "mcc": mcc,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        })


    results = pd.DataFrame(
        rows
    )

    # Primary selection criterion:
    # validation F1
    #
    # Tie breakers:
    # MCC -> ROC-AUC -> PR-AUC

    results = results.sort_values(
        by=[
            "f1",
            "mcc",
            "roc_auc",
            "pr_auc"
        ],
        ascending=False
    ).reset_index(
        drop=True
    )

    best = results.iloc[0]

    print(
        "\nBest validation weight:"
    )

    print(
        f"JIT weight     = "
        f"{best['weight_jit']:.2f}"
    )

    print(
        f"CodeBERT weight = "
        f"{best['weight_codebert']:.2f}"
    )

    print(
        f"Validation F1  = "
        f"{best['f1']:.4f}"
    )

    print(
        f"Validation MCC = "
        f"{best['mcc']:.4f}"
    )

    print(
        f"Validation ROC-AUC = "
        f"{best['roc_auc']:.4f}"
    )

    return (
        best,
        results
    )


# RF weight
best_rf_weight, rf_weight_results = search_best_weight(
    val_jit_rf_prob,
    val_codebert_rf_prob,
    y_val,
    "Random Forest"
)


# XGB weight
best_xgb_weight, xgb_weight_results = search_best_weight(
    val_jit_xgb_prob,
    val_codebert_xgb_prob,
    y_val,
    "XGBoost"
)


# ============================================================
# SAVE WEIGHT SEARCH
# ============================================================

rf_weight_results.to_csv(
    os.path.join(
        BLEND_DIR,
        "random_forest_validation_weights.csv"
    ),
    index=False
)

xgb_weight_results.to_csv(
    os.path.join(
        BLEND_DIR,
        "xgboost_validation_weights.csv"
    ),
    index=False
)


# ============================================================
# SELECTED WEIGHTS
# ============================================================

alpha_rf = float(
    best_rf_weight["weight_jit"]
)

alpha_xgb = float(
    best_xgb_weight["weight_jit"]
)


# ============================================================
# TEST PROBABILITIES
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "GENERATING TEST PROBABILITIES"
)

print(
    "=" * 80
)


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


# ============================================================
# APPLY FROZEN VALIDATION WEIGHTS
# ============================================================

test_rf_blended_probability = (
    alpha_rf * test_jit_rf_prob
    +
    (1.0 - alpha_rf)
    * test_codebert_rf_prob
)

test_xgb_blended_probability = (
    alpha_xgb * test_jit_xgb_prob
    +
    (1.0 - alpha_xgb)
    * test_codebert_xgb_prob
)


test_rf_prediction = (
    test_rf_blended_probability >= THRESHOLD
).astype(int)

test_xgb_prediction = (
    test_xgb_blended_probability >= THRESHOLD
).astype(int)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
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

        "experiment":
            "4E",

        "model":
            model_name,

        "accuracy":
            accuracy_score(
                y_true,
                predictions
            ),

        "precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "mcc":
            matthews_corrcoef(
                y_true,
                predictions
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                probabilities
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                probabilities
            ),

        "tn":
            int(cm[0, 0]),

        "fp":
            int(cm[0, 1]),

        "fn":
            int(cm[1, 0]),

        "tp":
            int(cm[1, 1]),
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
        f"Accuracy : "
        f"{results['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{results['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{results['recall']:.4f}"
    )

    print(
        f"F1       : "
        f"{results['f1']:.4f}"
    )

    print(
        f"MCC      : "
        f"{results['mcc']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{results['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC   : "
        f"{results['pr_auc']:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(cm)

    return results


rf_results = evaluate(
    y_test,
    test_rf_prediction,
    test_rf_blended_probability,
    "RF Probability Blend"
)

xgb_results = evaluate(
    y_test,
    test_xgb_prediction,
    test_xgb_blended_probability,
    "XGBoost Probability Blend"
)


# ============================================================
# SAVE TEST RESULTS
# ============================================================

results_df = pd.DataFrame([
    rf_results,
    xgb_results
])

results_df[
    "jit_weight"
] = [
    alpha_rf,
    alpha_xgb
]

results_df[
    "codebert_weight"
] = [
    1.0 - alpha_rf,
    1.0 - alpha_xgb
]

results_path = os.path.join(
    METRICS_DIR,
    "experiment_4e_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_predictions = pd.DataFrame({

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

    "rf_blended_probability":
        test_rf_blended_probability,

    "rf_blended_prediction":
        test_rf_prediction,

    "jit_xgb_probability":
        test_jit_xgb_prob,

    "codebert_xgb_probability":
        test_codebert_xgb_prob,

    "xgb_blended_probability":
        test_xgb_blended_probability,

    "xgb_blended_prediction":
        test_xgb_prediction,
})

predictions_path = os.path.join(
    PREDICTIONS_DIR,
    "test_probability_blend_predictions.csv"
)

test_predictions.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# SAVE BASE MODELS
# ============================================================

joblib.dump(
    jit_rf,
    os.path.join(
        MODELS_DIR,
        "jit_random_forest.joblib"
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
    jit_xgb,
    os.path.join(
        MODELS_DIR,
        "jit_xgboost.joblib"
    )
)

joblib.dump(
    codebert_xgb,
    os.path.join(
        MODELS_DIR,
        "codebert_xgboost.joblib"
    )
)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

config = {

    "experiment":
        "4E",

    "method":
        "Weighted Probability Blending",

    "formula":
        "P_blend = alpha * P_JIT + "
        "(1-alpha) * P_CodeBERT",

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

    "weight_selection":
        "validation F1",

    "test_weight_tuning":
        False,

    "rf_jit_weight":
        alpha_rf,

    "rf_codebert_weight":
        1.0 - alpha_rf,

    "xgb_jit_weight":
        alpha_xgb,

    "xgb_codebert_weight":
        1.0 - alpha_xgb,

    "threshold":
        THRESHOLD,

    "candidate_weights":
        [
            float(x)
            for x in BLEND_WEIGHTS
        ],

    "random_state":
        RANDOM_STATE,

    "split_strategy":
        "project-wise chronological 70/15/15",
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "experiment_4e_config.json"
    ),
    "w"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "EXPERIMENT 4E COMPLETED"
)

print(
    "=" * 80
)

print(
    "\nSelected validation weights:"
)

print(
    f"Random Forest:"
)

print(
    f"  JIT      = {alpha_rf:.2f}"
)

print(
    f"  CodeBERT = {1.0 - alpha_rf:.2f}"
)

print(
    f"\nXGBoost:"
)

print(
    f"  JIT      = {alpha_xgb:.2f}"
)

print(
    f"  CodeBERT = {1.0 - alpha_xgb:.2f}"
)


print(
    "\nFinal TEST results:"
)

print(
    results_df[
        [
            "model",
            "jit_weight",
            "codebert_weight",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "mcc",
            "roc_auc",
            "pr_auc",
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
    "\n" + "=" * 80
)

print(
    "ALL VERIFICATIONS PASSED"
)

print(
    "=" * 80
)