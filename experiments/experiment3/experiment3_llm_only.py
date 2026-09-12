import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
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
    precision_recall_curve
)

from xgboost import XGBClassifier

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "llm_experiment_preprocessed"

RESULTS_DIR = PROJECT_ROOT / "results" / "exp3"
MODELS_DIR = RESULTS_DIR / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA FILES
# ============================================================

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"

RANDOM_STATE = 42


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("EXPERIMENT 3 - LLM ONLY")
print("RANDOM FOREST + XGBOOST")
print("=" * 75)


# ============================================================
# READ DATA
# ============================================================

print("\nReading datasets...")

train_df = pd.read_csv(
    TRAIN_FILE,
    low_memory=False
)

validation_df = pd.read_csv(
    VALIDATION_FILE,
    low_memory=False
)

test_df = pd.read_csv(
    TEST_FILE,
    low_memory=False
)

print(f"Train      : {train_df.shape}")
print(f"Validation : {validation_df.shape}")
print(f"Test       : {test_df.shape}")


# ============================================================
# CHECK TARGET
# ============================================================

for name, df in [
    ("train", train_df),
    ("validation", validation_df),
    ("test", test_df)
]:

    if "buggy" not in df.columns:
        raise ValueError(
            f"'buggy' column not found in {name}.csv"
        )


# ============================================================
# SEPARATE X AND y
# ============================================================

X_train = train_df.drop(columns=["buggy"])
y_train = train_df["buggy"].astype(int)

X_validation = validation_df.drop(columns=["buggy"])
y_validation = validation_df["buggy"].astype(int)

X_test = test_df.drop(columns=["buggy"])
y_test = test_df["buggy"].astype(int)


# ============================================================
# FEATURE CONSISTENCY CHECK
# ============================================================

if list(X_train.columns) != list(X_validation.columns):
    raise ValueError(
        "Train and validation feature columns do not match."
    )

if list(X_train.columns) != list(X_test.columns):
    raise ValueError(
        "Train and test feature columns do not match."
    )


print(f"\nNumber of input features: {X_train.shape[1]}")


# ============================================================
# BUGGY DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("BUGGY DISTRIBUTION")
print("=" * 75)


def print_distribution(name, y):

    counts = y.value_counts().sort_index()
    percentages = (
        y.value_counts(normalize=True).sort_index() * 100
    )

    print(f"\n{name}:")

    for label in [0, 1]:

        count = counts.get(label, 0)
        percentage = percentages.get(label, 0)

        label_name = (
            "Non-buggy"
            if label == 0
            else "Buggy"
        )

        print(
            f"  {label_name}: "
            f"{count} ({percentage:.2f}%)"
        )


print_distribution("Train", y_train)
print_distribution("Validation", y_validation)
print_distribution("Test", y_test)


# ============================================================
# XGBOOST CLASS WEIGHT
# ============================================================

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

if positive_count == 0:
    raise ValueError(
        "No buggy samples found in training data."
    )

scale_pos_weight = (
    negative_count / positive_count
)

print("\n" + "=" * 75)
print("CLASS IMBALANCE")
print("=" * 75)

print(f"Negative samples: {negative_count}")
print(f"Positive samples: {positive_count}")
print(
    f"XGBoost scale_pos_weight: "
    f"{scale_pos_weight:.6f}"
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_probability
):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {

        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_true,
                y_probability
            )
        ),

        "pr_auc": float(
            average_precision_score(
                y_true,
                y_probability
            )
        ),

        "mcc": float(
            matthews_corrcoef(
                y_true,
                y_pred
            )
        ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


# ============================================================
# RANDOM FOREST
# ============================================================

print("\n" + "=" * 75)
print("TRAINING RANDOM FOREST")
print("=" * 75)

rf_model = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)

print("Random Forest training completed.")


# ============================================================
# RANDOM FOREST PREDICTIONS
# ============================================================

rf_validation_probability = (
    rf_model.predict_proba(X_validation)[:, 1]
)

rf_validation_prediction = (
    rf_validation_probability >= 0.5
).astype(int)


rf_test_probability = (
    rf_model.predict_proba(X_test)[:, 1]
)

rf_test_prediction = (
    rf_test_probability >= 0.5
).astype(int)


# ============================================================
# RANDOM FOREST METRICS
# ============================================================

rf_validation_metrics = calculate_metrics(
    y_validation,
    rf_validation_prediction,
    rf_validation_probability
)

rf_test_metrics = calculate_metrics(
    y_test,
    rf_test_prediction,
    rf_test_probability
)


# ============================================================
# XGBOOST
# ============================================================

print("\n" + "=" * 75)
print("TRAINING XGBOOST")
print("=" * 75)

xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method="hist"
)

xgb_model.fit(
    X_train,
    y_train
)

print("XGBoost training completed.")


# ============================================================
# XGBOOST PREDICTIONS
# ============================================================

xgb_validation_probability = (
    xgb_model.predict_proba(X_validation)[:, 1]
)

xgb_validation_prediction = (
    xgb_validation_probability >= 0.5
).astype(int)


xgb_test_probability = (
    xgb_model.predict_proba(X_test)[:, 1]
)

xgb_test_prediction = (
    xgb_test_probability >= 0.5
).astype(int)


# ============================================================
# XGBOOST METRICS
# ============================================================

xgb_validation_metrics = calculate_metrics(
    y_validation,
    xgb_validation_prediction,
    xgb_validation_probability
)

xgb_test_metrics = calculate_metrics(
    y_test,
    xgb_test_prediction,
    xgb_test_probability
)


# ============================================================
# SAVE MODEL METRICS
# ============================================================

rf_metrics = {
    "experiment": "experiment_3_llm_only",
    "model": "random_forest",
    "dataset_rows": (
        len(train_df) +
        len(validation_df) +
        len(test_df)
    ),
    "train_rows": len(train_df),
    "validation_rows": len(validation_df),
    "test_rows": len(test_df),
    "features": list(X_train.columns),
    "class_weight": "balanced",
    "validation": rf_validation_metrics,
    "test": rf_test_metrics
}


xgb_metrics = {
    "experiment": "experiment_3_llm_only",
    "model": "xgboost",
    "dataset_rows": (
        len(train_df) +
        len(validation_df) +
        len(test_df)
    ),
    "train_rows": len(train_df),
    "validation_rows": len(validation_df),
    "test_rows": len(test_df),
    "features": list(X_train.columns),
    "scale_pos_weight": float(
        scale_pos_weight
    ),
    "validation": xgb_validation_metrics,
    "test": xgb_test_metrics
}


with open(
    RESULTS_DIR / "random_forest_metrics.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        rf_metrics,
        f,
        indent=4
    )


with open(
    RESULTS_DIR / "xgboost_metrics.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        xgb_metrics,
        f,
        indent=4
    )


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {
    "experiment": "experiment_3_llm_only",

    "dataset": {
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
        "test_rows": len(test_df),

        "total_rows": (
            len(train_df) +
            len(validation_df) +
            len(test_df)
        ),

        "input_features": X_train.shape[1]
    },

    "class_distribution": {

        "train": {
            "non_buggy": int(
                (y_train == 0).sum()
            ),
            "buggy": int(
                (y_train == 1).sum()
            )
        },

        "validation": {
            "non_buggy": int(
                (y_validation == 0).sum()
            ),
            "buggy": int(
                (y_validation == 1).sum()
            )
        },

        "test": {
            "non_buggy": int(
                (y_test == 0).sum()
            ),
            "buggy": int(
                (y_test == 1).sum()
            )
        }
    },

    "xgboost_scale_pos_weight": float(
        scale_pos_weight
    ),

    "models": {

        "random_forest": {
            "validation": rf_validation_metrics,
            "test": rf_test_metrics
        },

        "xgboost": {
            "validation": xgb_validation_metrics,
            "test": xgb_test_metrics
        }
    }
}


with open(
    RESULTS_DIR / "summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# SPLIT REPORT
# ============================================================

split_report = {

    "experiment": "experiment_3_llm_only",

    "train_rows": len(train_df),
    "validation_rows": len(validation_df),
    "test_rows": len(test_df),

    "train_buggy_percentage": float(
        y_train.mean() * 100
    ),

    "validation_buggy_percentage": float(
        y_validation.mean() * 100
    ),

    "test_buggy_percentage": float(
        y_test.mean() * 100
    ),

    "feature_count": X_train.shape[1],

    "feature_names": list(
        X_train.columns
    ),

    "random_state": RANDOM_STATE,

    "threshold": 0.5
}


with open(
    RESULTS_DIR / "split_report.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        split_report,
        f,
        indent=4
    )


# ============================================================
# SAVE MODELS
# ============================================================

joblib.dump(
    rf_model,
    MODELS_DIR /
    "experiment_3_random_forest.joblib"
)

joblib.dump(
    xgb_model,
    MODELS_DIR /
    "experiment_3_xgboost.joblib"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(
    model,
    feature_names,
    model_name
):

    importance = pd.DataFrame({

        "feature": feature_names,

        "importance":
            model.feature_importances_

    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    importance.to_csv(
        RESULTS_DIR /
        f"{model_name}_feature_importance.csv",
        index=False
    )

    top_features = (
        importance
        .head(20)
        .sort_values("importance")
    )

    plt.figure(figsize=(10, 8))

    plt.barh(
        top_features["feature"],
        top_features["importance"]
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")

    plt.title(
        f"{model_name.replace('_', ' ').title()} "
        " - Top 20 Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR /
        f"{model_name}_feature_importance.png",
        dpi=300
    )

    plt.close()


save_feature_importance(
    rf_model,
    X_train.columns,
    "random_forest"
)

save_feature_importance(
    xgb_model,
    X_train.columns,
    "xgboost"
)


# ============================================================
# ROC AND PR CURVES
# ============================================================

def save_curves(
    y_true,
    probability,
    model_name
):

    fpr, tpr, _ = roc_curve(
        y_true,
        probability
    )

    precision, recall, _ = (
        precision_recall_curve(
            y_true,
            probability
        )
    )

    roc_auc = roc_auc_score(
        y_true,
        probability
    )

    pr_auc = average_precision_score(
        y_true,
        probability
    )


    # --------------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {roc_auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        f"{model_name} - ROC Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR /
        f"{model_name.lower().replace(' ', '_')}_roc.png",
        dpi=300
    )

    plt.close()


    # --------------------------------------------------------
    # PRECISION-RECALL CURVE
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.plot(
        recall,
        precision,
        label=f"PR-AUC = {pr_auc:.4f}"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title(
        f"{model_name} - Precision-Recall Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR /
        f"{model_name.lower().replace(' ', '_')}_pr.png",
        dpi=300
    )

    plt.close()


save_curves(
    y_test,
    rf_test_probability,
    "Random Forest"
)

save_curves(
    y_test,
    xgb_test_probability,
    "XGBoost"
)


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    model_name,
    validation_metrics,
    test_metrics
):

    print("\n" + "=" * 75)
    print(model_name.upper())
    print("=" * 75)

    print("\nVALIDATION:")

    for key, value in validation_metrics.items():

        print(
            f"  {key:10s}: {value}"
        )

    print("\nTEST:")

    for key, value in test_metrics.items():

        print(
            f"  {key:10s}: {value}"
        )


print_metrics(
    "Random Forest",
    rf_validation_metrics,
    rf_test_metrics
)

print_metrics(
    "XGBoost",
    xgb_validation_metrics,
    xgb_test_metrics
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 75)
print("EXPERIMENT 3 COMPLETED")
print("=" * 75)

print(f"\nResults saved to:")
print(RESULTS_DIR)

print("\nModels saved to:")
print(MODELS_DIR)

print("\n" + "=" * 75)