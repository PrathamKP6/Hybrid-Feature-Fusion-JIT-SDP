"""
Experiment 4C — Mutual Information Top-25 Feature Selection

Hybrid JIT + CodeBERT Feature Fusion
-------------------------------------

Pipeline:
    12 JIT features
        +
    384 PCA-reduced CodeBERT features
        |
        v
    396 candidate features
        |
        v
    Mutual Information feature ranking
        |
        v
    Select TOP 25 features
        |
        v
    Random Forest / XGBoost
        |
        v
    Validation + Test evaluation

IMPORTANT:
- Project-wise chronological 70/15/15 split is already completed.
- PCA was fitted on TRAIN ONLY in the previous pipeline.
- Mutual Information is fitted on TRAIN ONLY.
- No resampling.
- No random train/test split.
- Test set is used only for final evaluation.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif
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

import matplotlib.pyplot as plt

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\Hybrid-Feature-Fusion-JIT-SDP"

# Original chronological splits containing JIT features
JIT_SPLIT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "chronological_splits"
)

# PCA-reduced CodeBERT data
PCA_DIR = os.path.join(
    BASE_DIR,
    "results",
    "pca"
)

TRAIN_JIT = os.path.join(JIT_SPLIT_DIR, "train.csv")
VAL_JIT = os.path.join(JIT_SPLIT_DIR, "validation.csv")
TEST_JIT = os.path.join(JIT_SPLIT_DIR, "test.csv")

TRAIN_PCA = os.path.join(PCA_DIR, "train_pca384.csv")
VAL_PCA = os.path.join(PCA_DIR, "validation_pca384.csv")
TEST_PCA = os.path.join(PCA_DIR, "test_pca384.csv")


# Output directory
OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "experiment_4c_mi_top25"
)

METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
FEATURE_DIR = os.path.join(OUTPUT_DIR, "feature_selection")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

for directory in [
    OUTPUT_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    FEATURE_DIR,
    PLOTS_DIR,
]:
    os.makedirs(directory, exist_ok=True)


# ============================================================
# EXPERIMENT PARAMETERS
# ============================================================

RANDOM_STATE = 42

N_TOTAL_CANDIDATE_FEATURES = 396
N_SELECTED_FEATURES = 25

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

N_JIT_FEATURES = len(JIT_FEATURES)

PCA_DIMENSION = 384

assert N_JIT_FEATURES + PCA_DIMENSION == N_TOTAL_CANDIDATE_FEATURES


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_target(series):
    """
    Convert buggy labels into integer 0/1.
    """

    def convert(value):

        if pd.isna(value):
            return np.nan

        if isinstance(value, bool):
            return int(value)

        if isinstance(value, (int, np.integer)):
            return int(value)

        if isinstance(value, (float, np.floating)):
            return int(value)

        value_str = str(value).strip().lower()

        if value_str in {"1", "true", "yes", "buggy"}:
            return 1

        if value_str in {"0", "false", "no", "clean", "non-buggy"}:
            return 0

        raise ValueError(
            f"Unable to normalize target value: {value}"
        )

    return series.apply(convert).astype(int)


def parse_embedding(value):

    if isinstance(value, list):
        return np.asarray(value, dtype=np.float32)

    if isinstance(value, np.ndarray):
        return value.astype(np.float32)

    if pd.isna(value):
        return None

    try:
        parsed = json.loads(value)

        if isinstance(parsed, list):
            return np.asarray(parsed, dtype=np.float32)

    except Exception:
        pass

    return None


def extract_pca_features(df):
    """
    Extract 384-dimensional PCA embedding.
    """

    vectors = []

    for value in df["embedding"]:

        vector = parse_embedding(value)

        if vector is None:
            raise ValueError(
                "Invalid embedding encountered."
            )

        if len(vector) != PCA_DIMENSION:
            raise ValueError(
                f"Expected {PCA_DIMENSION}-D embedding, "
                f"found {len(vector)} dimensions."
            )

        vectors.append(vector)

    return np.vstack(vectors)


def evaluate_model(model, X, y, split_name, model_name):

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    cm = confusion_matrix(y, predictions)

    results = {
        "experiment": "4C",
        "split": split_name,
        "model": model_name,

        "n_features": X.shape[1],

        "accuracy": accuracy_score(
            y,
            predictions
        ),

        "precision": precision_score(
            y,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            y,
            predictions,
            zero_division=0
        ),

        "mcc": matthews_corrcoef(
            y,
            predictions
        ),

        "roc_auc": roc_auc_score(
            y,
            probabilities
        ),

        "pr_auc": average_precision_score(
            y,
            probabilities
        ),

        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    print("\n" + "=" * 70)
    print(f"{model_name} — {split_name}")
    print("=" * 70)

    print(f"Accuracy : {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall   : {results['recall']:.4f}")
    print(f"F1       : {results['f1']:.4f}")
    print(f"MCC      : {results['mcc']:.4f}")
    print(f"ROC-AUC  : {results['roc_auc']:.4f}")
    print(f"PR-AUC   : {results['pr_auc']:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    return results, predictions, probabilities



# ============================================================
# PLOT ROC AND PRECISION-RECALL CURVES
# ============================================================

def plot_roc_pr(
    y_true,
    probabilities,
    model_name,
    split_name
):
    """
    Save ROC and Precision-Recall curves.

    PR-AUC uses average_precision_score, matching the
    PR-AUC metric reported by evaluate_model().
    """

    # --------------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------------

    fpr, tpr, _ = roc_curve(
        y_true,
        probabilities
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=(
            f"{model_name} "
            f"(ROC-AUC = {roc_auc:.4f})"
        )
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Random classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        f"ROC Curve - {model_name} - "
        f"{split_name.title()}"
    )

    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    roc_path = os.path.join(
        PLOTS_DIR,
        f"{model_name.lower().replace(' ', '_')}_"
        f"{split_name}_roc.png"
    )

    plt.savefig(
        roc_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # PRECISION-RECALL CURVE
    # --------------------------------------------------------

    precision, recall, _ = precision_recall_curve(
        y_true,
        probabilities
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        recall,
        precision,
        linewidth=2,
        label=(
            f"{model_name} "
            f"(PR-AUC = {pr_auc:.4f})"
        )
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title(
        f"Precision-Recall Curve - {model_name} - "
        f"{split_name.title()}"
    )

    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    pr_path = os.path.join(
        PLOTS_DIR,
        f"{model_name.lower().replace(' ', '_')}_"
        f"{split_name}_pr.png"
    )

    plt.savefig(
        pr_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"ROC curve saved: {roc_path}")
    print(f"PR curve saved:  {pr_path}")

    return roc_path, pr_path


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 80)
print("EXPERIMENT 4C — MI TOP-25 FEATURE SELECTION")
print("=" * 80)

print("\nLoading JIT datasets...")

train_jit = pd.read_csv(TRAIN_JIT)
val_jit = pd.read_csv(VAL_JIT)
test_jit = pd.read_csv(TEST_JIT)

print(f"Train JIT: {len(train_jit):,}")
print(f"Val JIT  : {len(val_jit):,}")
print(f"Test JIT : {len(test_jit):,}")


print("\nLoading PCA datasets...")

train_pca = pd.read_csv(TRAIN_PCA)
val_pca = pd.read_csv(VAL_PCA)
test_pca = pd.read_csv(TEST_PCA)

print(f"Train PCA: {len(train_pca):,}")
print(f"Val PCA  : {len(val_pca):,}")
print(f"Test PCA : {len(test_pca):,}")


# ============================================================
# BASIC DATA VALIDATION
# ============================================================

print("\nValidating commit IDs...")

assert train_jit["commit_id"].is_unique
assert val_jit["commit_id"].is_unique
assert test_jit["commit_id"].is_unique

assert train_pca["commit_id"].is_unique
assert val_pca["commit_id"].is_unique
assert test_pca["commit_id"].is_unique


# ============================================================
# MERGE JIT + CODEBERT
# ============================================================

def build_hybrid_dataset(jit_df, pca_df, split_name):

    print(f"\nBuilding {split_name} hybrid dataset...")

    jit = jit_df[
        ["commit_id", "project", "buggy"] + JIT_FEATURES
    ].copy()

    pca = pca_df[
        ["commit_id", "project", "buggy", "embedding"]
    ].copy()

    # Normalize target independently
    jit["buggy"] = normalize_target(jit["buggy"])
    pca["buggy"] = normalize_target(pca["buggy"])

    # Validate project/label consistency BEFORE dropping them
    check = jit[
        ["commit_id", "project", "buggy"]
    ].merge(
        pca[
            ["commit_id", "project", "buggy"]
        ],
        on="commit_id",
        suffixes=("_jit", "_pca"),
        how="inner"
    )

    assert len(check) == len(jit), (
        f"{split_name}: commit mismatch during merge."
    )

    assert (
        check["project_jit"] ==
        check["project_pca"]
    ).all(), (
        f"{split_name}: project mismatch detected."
    )

    assert (
        check["buggy_jit"] ==
        check["buggy_pca"]
    ).all(), (
        f"{split_name}: buggy label mismatch detected."
    )

    # Merge only on commit_id
    merged = jit.merge(
        pca[
            ["commit_id", "embedding"]
        ],
        on="commit_id",
        how="inner",
        validate="one_to_one"
    )

    print(
        f"{split_name}: merged rows = "
        f"{len(merged):,}"
    )

    return merged


train = build_hybrid_dataset(
    train_jit,
    train_pca,
    "TRAIN"
)

val = build_hybrid_dataset(
    val_jit,
    val_pca,
    "VALIDATION"
)

test = build_hybrid_dataset(
    test_jit,
    test_pca,
    "TEST"
)


# ============================================================
# BUILD 396-D CANDIDATE FEATURE MATRIX
# ============================================================

print("\n" + "=" * 80)
print("BUILDING 396-D HYBRID FEATURE SPACE")
print("=" * 80)


def build_feature_matrix(df):

    # JIT
    jit_values = df[JIT_FEATURES].copy()

    # Convert to numeric
    for feature in JIT_FEATURES:
        jit_values[feature] = pd.to_numeric(
            jit_values[feature],
            errors="coerce"
        )

    return jit_values


X_train_jit = build_feature_matrix(train)
X_val_jit = build_feature_matrix(val)
X_test_jit = build_feature_matrix(test)


# ============================================================
# TRAIN-ONLY JIT IMPUTATION
# ============================================================

print("\nApplying train-only JIT median imputation...")

jit_medians = X_train_jit.median()

X_train_jit = X_train_jit.fillna(jit_medians)
X_val_jit = X_val_jit.fillna(jit_medians)
X_test_jit = X_test_jit.fillna(jit_medians)


# ============================================================
# EXTRACT 384-D CODEBERT PCA FEATURES
# ============================================================

print("\nExtracting PCA embeddings...")

X_train_pca = extract_pca_features(train)
X_val_pca = extract_pca_features(val)
X_test_pca = extract_pca_features(test)

print(
    "PCA shapes:",
    X_train_pca.shape,
    X_val_pca.shape,
    X_test_pca.shape
)


# ============================================================
# CREATE FEATURE NAMES
# ============================================================

pca_feature_names = [
    f"pca_{i}"
    for i in range(1, PCA_DIMENSION + 1)
]

all_feature_names = (
    JIT_FEATURES +
    pca_feature_names
)

assert len(all_feature_names) == 396


# ============================================================
# COMBINE JIT + CODEBERT
# ============================================================

X_train = np.hstack([
    X_train_jit.values,
    X_train_pca
])

X_val = np.hstack([
    X_val_jit.values,
    X_val_pca
])

X_test = np.hstack([
    X_test_jit.values,
    X_test_pca
])

print("\nFinal candidate feature shapes:")

print("Train:", X_train.shape)
print("Val  :", X_val.shape)
print("Test :", X_test.shape)

assert X_train.shape[1] == 396
assert X_val.shape[1] == 396
assert X_test.shape[1] == 396


# ============================================================
# TARGET
# ============================================================

y_train = normalize_target(train["buggy"])
y_val = normalize_target(val["buggy"])
y_test = normalize_target(test["buggy"])

print("\nTarget distribution:")

print(
    f"Train: 0={sum(y_train == 0):,}, "
    f"1={sum(y_train == 1):,}"
)

print(
    f"Val  : 0={sum(y_val == 0):,}, "
    f"1={sum(y_val == 1):,}"
)

print(
    f"Test : 0={sum(y_test == 0):,}, "
    f"1={sum(y_test == 1):,}"
)


# ============================================================
# SANITY CHECK
# ============================================================

assert np.isfinite(X_train).all()
assert np.isfinite(X_val).all()
assert np.isfinite(X_test).all()


# ============================================================
# MUTUAL INFORMATION FEATURE SELECTION
# ============================================================

print("\n" + "=" * 80)
print("MUTUAL INFORMATION FEATURE SELECTION")
print("=" * 80)

print(
    "\nFitting Mutual Information ONLY on TRAINING DATA..."
)

mi_scores = mutual_info_classif(
    X_train,
    y_train,
    discrete_features=False,
    random_state=RANDOM_STATE,
    n_neighbors=3
)


# ============================================================
# RANK ALL FEATURES
# ============================================================

ranking = pd.DataFrame({
    "feature": all_feature_names,
    "mi_score": mi_scores,
    "feature_type": [
        "JIT" if name in JIT_FEATURES else "CodeBERT_PCA"
        for name in all_feature_names
    ]
})

ranking = ranking.sort_values(
    by="mi_score",
    ascending=False
).reset_index(drop=True)

ranking["rank"] = np.arange(
    1,
    len(ranking) + 1
)

ranking = ranking[
    [
        "rank",
        "feature",
        "feature_type",
        "mi_score"
    ]
]


# ============================================================
# SAVE COMPLETE MI RANKING
# ============================================================

ranking_path = os.path.join(
    FEATURE_DIR,
    "all_396_features_mi_ranking.csv"
)

ranking.to_csv(
    ranking_path,
    index=False
)


# ============================================================
# SELECT TOP 25
# ============================================================

top_features_df = ranking.head(
    N_SELECTED_FEATURES
).copy()

selected_features = top_features_df[
    "feature"
].tolist()

selected_indices = [
    all_feature_names.index(feature)
    for feature in selected_features
]


print("\n" + "=" * 80)
print("TOP 25 SELECTED FEATURES")
print("=" * 80)

print(
    top_features_df.to_string(
        index=False
    )
)


# ============================================================
# FEATURE TYPE SUMMARY
# ============================================================

selected_jit = [
    feature
    for feature in selected_features
    if feature in JIT_FEATURES
]

selected_codebert = [
    feature
    for feature in selected_features
    if feature not in JIT_FEATURES
]

print("\nSelected feature composition:")
print(
    f"JIT features      : {len(selected_jit)}"
)

print(
    f"CodeBERT PCA      : {len(selected_codebert)}"
)

print(
    f"Total selected    : "
    f"{len(selected_features)}"
)


# ============================================================
# SAVE SELECTED FEATURE LIST
# ============================================================

selected_path = os.path.join(
    FEATURE_DIR,
    "top25_selected_features.csv"
)

top_features_df.to_csv(
    selected_path,
    index=False
)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

feature_config = {
    "experiment": "4C",
    "method": "Mutual Information",
    "candidate_features": 396,
    "selected_features": 25,
    "jit_features_total": 12,
    "codebert_pca_features_total": 384,
    "selected_jit_features": len(selected_jit),
    "selected_codebert_features": len(selected_codebert),
    "selected_feature_names": selected_features,
    "selection_fitted_on": "train_only",
    "random_state": RANDOM_STATE,
    "pca_dimension": 384,
    "pca_fitted_on": "train_only",
    "resampling": False,
    "split_strategy": "project-wise chronological 70/15/15",
}

with open(
    os.path.join(
        FEATURE_DIR,
        "feature_selection_config.json"
    ),
    "w"
) as f:

    json.dump(
        feature_config,
        f,
        indent=4
    )


# ============================================================
# APPLY TOP-25 FEATURES
# ============================================================

print("\nApplying selected features...")

X_train_selected = X_train[
    :,
    selected_indices
]

X_val_selected = X_val[
    :,
    selected_indices
]

X_test_selected = X_test[
    :,
    selected_indices
]


print(
    "\nSelected feature shapes:"
)

print(
    "Train:",
    X_train_selected.shape
)

print(
    "Val  :",
    X_val_selected.shape
)

print(
    "Test :",
    X_test_selected.shape
)

assert X_train_selected.shape[1] == 25
assert X_val_selected.shape[1] == 25
assert X_test_selected.shape[1] == 25


# ============================================================
# RANDOM FOREST
# ============================================================

print("\n" + "=" * 80)
print("TRAINING RANDOM FOREST")
print("=" * 80)

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_model.fit(
    X_train_selected,
    y_train
)

print("Random Forest training completed.")


# ============================================================
# XGBOOST
# ============================================================

print("\n" + "=" * 80)
print("TRAINING XGBOOST")
print("=" * 80)

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
    f"scale_pos_weight = "
    f"{scale_pos_weight:.4f}"
)

xgb_model = XGBClassifier(
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

xgb_model.fit(
    X_train_selected,
    y_train
)

print("XGBoost training completed.")


# ============================================================
# EVALUATION
# ============================================================

all_results = []


# ------------------------------
# Random Forest
# ------------------------------

rf_val_results, rf_val_pred, rf_val_prob = evaluate_model(
    rf_model,
    X_val_selected,
    y_val,
    "validation",
    "Random Forest"
)

rf_test_results, rf_test_pred, rf_test_prob = evaluate_model(
    rf_model,
    X_test_selected,
    y_test,
    "test",
    "Random Forest"
)

plot_roc_pr(
    y_val,
    rf_val_prob,
    "Random Forest",
    "validation"
)

plot_roc_pr(
    y_test,
    rf_test_prob,
    "Random Forest",
    "test"
)

all_results.extend([
    rf_val_results,
    rf_test_results
])


# ------------------------------
# XGBoost
# ------------------------------

xgb_val_results, xgb_val_pred, xgb_val_prob = evaluate_model(
    xgb_model,
    X_val_selected,
    y_val,
    "validation",
    "XGBoost"
)

xgb_test_results, xgb_test_pred, xgb_test_prob = evaluate_model(
    xgb_model,
    X_test_selected,
    y_test,
    "test",
    "XGBoost"
)

plot_roc_pr(
    y_val,
    xgb_val_prob,
    "XGBoost",
    "validation"
)

plot_roc_pr(
    y_test,
    xgb_test_prob,
    "XGBoost",
    "test"
)

all_results.extend([
    xgb_val_results,
    xgb_test_results
])


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame(
    all_results
)

metrics_path = os.path.join(
    METRICS_DIR,
    "experiment_4c_results.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_predictions = pd.DataFrame({
    "commit_id": test["commit_id"],
    "project": test["project"],
    "actual_buggy": y_test,

    "rf_prediction": rf_test_pred,
    "rf_probability": rf_test_prob,

    "xgb_prediction": xgb_test_pred,
    "xgb_probability": xgb_test_prob,
})

test_predictions.to_csv(
    os.path.join(
        PREDICTIONS_DIR,
        "test_predictions.csv"
    ),
    index=False
)


# ============================================================
# SAVE VALIDATION PREDICTIONS
# ============================================================

val_predictions = pd.DataFrame({
    "commit_id": val["commit_id"],
    "project": val["project"],
    "actual_buggy": y_val,

    "rf_prediction": rf_val_pred,
    "rf_probability": rf_val_prob,

    "xgb_prediction": xgb_val_pred,
    "xgb_probability": xgb_val_prob,
})

val_predictions.to_csv(
    os.path.join(
        PREDICTIONS_DIR,
        "validation_predictions.csv"
    ),
    index=False
)


# ============================================================
# SAVE MODELS
# ============================================================

import joblib

joblib.dump(
    rf_model,
    os.path.join(
        MODELS_DIR,
        "random_forest_top25_mi.joblib"
    )
)

joblib.dump(
    xgb_model,
    os.path.join(
        MODELS_DIR,
        "xgboost_top25_mi.joblib"
    )
)


# ============================================================
# SAVE CONFUSION MATRICES
# ============================================================

for model_name, split_name, cm in [
    (
        "random_forest",
        "validation",
        confusion_matrix(y_val, rf_val_pred)
    ),
    (
        "random_forest",
        "test",
        confusion_matrix(y_test, rf_test_pred)
    ),
    (
        "xgboost",
        "validation",
        confusion_matrix(y_val, xgb_val_pred)
    ),
    (
        "xgboost",
        "test",
        confusion_matrix(y_test, xgb_test_pred)
    ),
]:

    cm_df = pd.DataFrame(
        cm,
        index=["Actual_0", "Actual_1"],
        columns=["Predicted_0", "Predicted_1"]
    )

    cm_df.to_csv(
        os.path.join(
            METRICS_DIR,
            f"{model_name}_{split_name}_confusion_matrix.csv"
        )
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("EXPERIMENT 4C COMPLETED")
print("=" * 80)

print(
    "\nFeature reduction:"
)

print(
    f"396 candidate features → "
    f"{N_SELECTED_FEATURES} selected features"
)

print(
    f"Dimensionality reduction: "
    f"{(1 - N_SELECTED_FEATURES / 396) * 100:.2f}%"
)

print("\nSelected JIT features:")

if selected_jit:
    for feature in selected_jit:
        print(
            f"  {feature}"
        )
else:
    print("  None")


print("\nSelected CodeBERT PCA features:")

for feature in selected_codebert:
    print(
        f"  {feature}"
    )


print("\nTest results:")

test_results_df = metrics_df[
    metrics_df["split"] == "test"
][
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
]

print(
    test_results_df.to_string(
        index=False
    )
)


print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nFeature ranking:")
print(ranking_path)

print("\nSelected feature list:")
print(selected_path)

print("\nMetrics:")
print(metrics_path)

print("\nPlots:")
print(PLOTS_DIR)

print("\nGenerated plots:")
print("  random_forest_validation_roc.png")
print("  random_forest_validation_pr.png")
print("  random_forest_test_roc.png")
print("  random_forest_test_pr.png")
print("  xgboost_validation_roc.png")
print("  xgboost_validation_pr.png")
print("  xgboost_test_roc.png")
print("  xgboost_test_pr.png")

print("\n" + "=" * 80)
print("ALL VERIFICATIONS PASSED")
print("=" * 80)