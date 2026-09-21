"""
EXPERIMENT 8: JIT + CodeBERT + REDUCED LLM FEATURES

Purpose
-------
Evaluate whether a compact LLM representation based only on
confidence and margin features can preserve/improve predictive
performance compared with:

    Experiment 4F:
        JIT + 384 CodeBERT

    Experiment 7F:
        JIT + 384 CodeBERT + all 51 LLM features

Feature configuration
---------------------
JIT:
    12 features

CodeBERT:
    384 PCA components

LLM:
    14 confidence/margin features

Total:
    12 + 384 + 14 = 410 model features

Methodology
-----------
- Project-wise chronological 70/15/15 split
- Canonical buggy labels are authoritative
- PCA already fitted on TRAIN only
- LLM categorical encoding is NOT required because only
  confidence/margin features are used
- Median imputation fitted on TRAIN only
- No resampling
- No random split
- Threshold = 0.5
- RF/XGBoost/LightGBM
- Validation-only blend weight selection
- Test evaluated only after model/weight selection

Models
------
RF:
    n_estimators=300
    max_features="sqrt"
    class_weight="balanced"

XGBoost:
    n_estimators=300
    max_depth=6
    learning_rate=0.05
    subsample=0.8
    colsample_bytree=0.8
    scale_pos_weight=train_neg/train_pos

LightGBM:
    n_estimators=300
    max_depth=6
    learning_rate=0.05
    subsample=0.8
    colsample_bytree=0.8
    scale_pos_weight=train_neg/train_pos

Blend:
    RF + XGB + LGBM
    validation-selected weights
    grid step = 0.05

Selection priority:
    F1 -> MCC -> ROC-AUC -> PR-AUC
"""

import os
import json
import ast
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = r"D:\Hybrid-Feature-Fusion-JIT-SDP"

JIT_DIR = os.path.join(ROOT, "results", "chronological_splits")
PCA_DIR = os.path.join(ROOT, "results", "pca")
LLM_DIR = os.path.join(ROOT, "data", "llm_experiment_dataset")

OUT_DIR = os.path.join(
    ROOT,
    "results",
    "experiment_8_jit_codebert_reduced_llm"
)

PLOTS_DIR = os.path.join(OUT_DIR, "plots")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ============================================================
# FEATURES
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

LLM_REDUCED_FEATURES = [
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
ID_COL = "commit_id"

EXPECTED_TOTAL_FEATURES = 12 + 384 + 14


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def normalize_target(series):
    """
    Convert buggy labels to integer 0/1.
    """

    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype(int)

    mapping = {
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
        "yes": 1,
        "no": 0,
        "buggy": 1,
        "non-buggy": 0,
        "non_buggy": 0,
        "nonbuggy": 0,
    }

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
        .astype(int)
    )


def parse_embedding(value):
    """
    Parse a JSON/Python-list embedding representation.
    """

    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=np.float32)

    if pd.isna(value):
        return None

    text = str(value).strip()

    try:
        return np.asarray(json.loads(text), dtype=np.float32)
    except Exception:
        try:
            return np.asarray(ast.literal_eval(text), dtype=np.float32)
        except Exception:
            raise ValueError(
                f"Could not parse embedding value: {text[:100]}"
            )


def load_pca_file(path):
    """
    Load PCA-reduced chronological split.

    The embedding column contains the 384-dimensional
    PCA representation.
    """

    df = pd.read_csv(path)

    embeddings = np.vstack(
        df["embedding"].apply(parse_embedding).values
    )

    if embeddings.shape[1] != 384:
        raise ValueError(
            f"{path}: expected 384-D embeddings, "
            f"found {embeddings.shape[1]}"
        )

    return df, embeddings


def load_llm_dataset():
    """
    Load the LLM dataset.

    Supports one CSV or multiple CSV files inside
    data/llm_experiment_dataset.
    """

    csv_files = [
        os.path.join(LLM_DIR, f)
        for f in os.listdir(LLM_DIR)
        if f.lower().endswith(".csv")
    ]

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {LLM_DIR}"
        )

    frames = []

    for path in sorted(csv_files):
        print(f"Loading LLM file: {path}")
        frames.append(pd.read_csv(path))

    llm = pd.concat(
        frames,
        ignore_index=True
    )

    # Remove accidental duplicate rows by commit_id
    if ID_COL not in llm.columns:
        raise ValueError(
            "LLM dataset does not contain commit_id"
        )

    if llm[ID_COL].duplicated().any():
        dup_count = llm[ID_COL].duplicated().sum()

        print(
            f"WARNING: {dup_count} duplicate LLM commit IDs found."
        )

        llm = llm.drop_duplicates(
            subset=[ID_COL],
            keep="first"
        )

    missing = [
        c for c in LLM_REDUCED_FEATURES
        if c not in llm.columns
    ]

    if missing:
        raise ValueError(
            f"Missing LLM features:\n{missing}"
        )

    return llm[
        [ID_COL] + LLM_REDUCED_FEATURES
    ].copy()


def metrics(y_true, probabilities):
    predictions = (
        probabilities >= 0.5
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        predictions
    )

    return {
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
        "confusion_matrix": cm.tolist(),
    }


def print_metrics(name, result):
    print(f"\n{name}")
    print("-" * 70)

    print(
        f"Accuracy : {result['accuracy']:.6f}"
    )
    print(
        f"Precision: {result['precision']:.6f}"
    )
    print(
        f"Recall   : {result['recall']:.6f}"
    )
    print(
        f"F1       : {result['f1']:.6f}"
    )
    print(
        f"MCC      : {result['mcc']:.6f}"
    )
    print(
        f"ROC-AUC  : {result['roc_auc']:.6f}"
    )
    print(
        f"PR-AUC   : {result['pr_auc']:.6f}"
    )

    print(
        "CM       :",
        np.array(result["confusion_matrix"])
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("EXPERIMENT 8: JIT + CODEBERT + REDUCED LLM")
print("=" * 80)

print("\nFeature configuration:")
print(f"  JIT       : {len(JIT_FEATURES)}")
print("  CodeBERT  : 384")
print(f"  LLM       : {len(LLM_REDUCED_FEATURES)}")
print(
    f"  TOTAL     : {EXPECTED_TOTAL_FEATURES}"
)


print("\nLoading chronological JIT splits...")

train_jit = pd.read_csv(
    os.path.join(
        JIT_DIR,
        "train.csv"
    )
)

val_jit = pd.read_csv(
    os.path.join(
        JIT_DIR,
        "validation.csv"
    )
)

test_jit = pd.read_csv(
    os.path.join(
        JIT_DIR,
        "test.csv"
    )
)

print(
    f"Train: {len(train_jit):,}"
)
print(
    f"Validation: {len(val_jit):,}"
)
print(
    f"Test: {len(test_jit):,}"
)


# ============================================================
# LOAD PCA CODEBERT
# ============================================================

print("\nLoading PCA CodeBERT representations...")

train_pca_df, train_cb = load_pca_file(
    os.path.join(
        PCA_DIR,
        "train_pca384.csv"
    )
)

val_pca_df, val_cb = load_pca_file(
    os.path.join(
        PCA_DIR,
        "validation_pca384.csv"
    )
)

test_pca_df, test_cb = load_pca_file(
    os.path.join(
        PCA_DIR,
        "test_pca384.csv"
    )
)

print(
    "CodeBERT dimensions:",
    train_cb.shape,
    val_cb.shape,
    test_cb.shape
)


# ============================================================
# LOAD LLM
# ============================================================

print("\nLoading reduced LLM features...")

llm = load_llm_dataset()

print(
    f"LLM rows: {len(llm):,}"
)

print(
    f"LLM features: {len(LLM_REDUCED_FEATURES)}"
)


# ============================================================
# ALIGN DATA BY COMMIT ID
# ============================================================

print("\nAligning LLM features using commit_id...")

llm_indexed = llm.set_index(
    ID_COL
)

def build_split(jit_df, cb_df, cb_matrix, split_name):

    ids_jit = jit_df[ID_COL].astype(str)
    ids_cb = cb_df[ID_COL].astype(str)

    if not np.array_equal(
        ids_jit.values,
        ids_cb.values
    ):
        raise ValueError(
            f"{split_name}: JIT and PCA commit IDs "
            "are not in identical order."
        )

    if ids_jit.duplicated().any():
        raise ValueError(
            f"{split_name}: duplicate commit IDs in JIT."
        )

    # LLM alignment
    split_llm = llm_indexed.reindex(
        ids_jit
    )

    if split_llm.isna().all(axis=None):
        raise ValueError(
            f"{split_name}: LLM alignment failed."
        )

    missing_count = split_llm.isna().any(axis=1).sum()

    if missing_count > 0:
        raise ValueError(
            f"{split_name}: "
            f"{missing_count} commit IDs missing LLM features."
        )

    # Verify labels against canonical JIT labels
    y = normalize_target(
        jit_df[TARGET]
    ).to_numpy()

    print(
        f"{split_name}: "
        f"rows={len(jit_df):,}, "
        f"buggy={y.sum():,}, "
        f"LLM matched={len(split_llm):,}"
    )

    return (
        jit_df,
        cb_matrix,
        split_llm.reset_index(drop=True),
        y,
    )


train_jit, train_cb, train_llm, y_train = build_split(
    train_jit,
    train_pca_df,
    train_cb,
    "TRAIN"
)

val_jit, val_cb, val_llm, y_val = build_split(
    val_jit,
    val_pca_df,
    val_cb,
    "VALIDATION"
)

test_jit, test_cb, test_llm, y_test = build_split(
    test_jit,
    test_pca_df,
    test_cb,
    "TEST"
)


# ============================================================
# BUILD JIT MATRICES
# ============================================================

def get_jit_matrix(df):
    missing = [
        c for c in JIT_FEATURES
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing JIT features: {missing}"
        )

    return df[
        JIT_FEATURES
    ].apply(
        pd.to_numeric,
        errors="coerce"
    ).to_numpy(
        dtype=np.float32
    )


X_train_jit = get_jit_matrix(train_jit)
X_val_jit = get_jit_matrix(val_jit)
X_test_jit = get_jit_matrix(test_jit)


# ============================================================
# LLM NUMERICAL CONVERSION
# ============================================================

def get_llm_matrix(df):
    return df[
        LLM_REDUCED_FEATURES
    ].apply(
        pd.to_numeric,
        errors="coerce"
    ).to_numpy(
        dtype=np.float32
    )


X_train_llm = get_llm_matrix(train_llm)
X_val_llm = get_llm_matrix(val_llm)
X_test_llm = get_llm_matrix(test_llm)


# ============================================================
# TRAIN-ONLY IMPUTATION
# ============================================================

print("\nFitting median imputer on TRAIN only...")

imputer = SimpleImputer(
    strategy="median"
)

X_train_llm = imputer.fit_transform(
    X_train_llm
).astype(np.float32)

X_val_llm = imputer.transform(
    X_val_llm
).astype(np.float32)

X_test_llm = imputer.transform(
    X_test_llm
).astype(np.float32)


# ============================================================
# FINAL FEATURE MATRICES
# ============================================================

X_train = np.hstack([
    X_train_jit,
    train_cb,
    X_train_llm,
]).astype(np.float32)

X_val = np.hstack([
    X_val_jit,
    val_cb,
    X_val_llm,
]).astype(np.float32)

X_test = np.hstack([
    X_test_jit,
    test_cb,
    X_test_llm,
]).astype(np.float32)


print("\nFinal feature matrices:")
print(
    f"Train: {X_train.shape}"
)
print(
    f"Validation: {X_val.shape}"
)
print(
    f"Test: {X_test.shape}"
)

assert X_train.shape[1] == 410
assert X_val.shape[1] == 410
assert X_test.shape[1] == 410


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\nLabel distribution:")

for name, y in [
    ("Train", y_train),
    ("Validation", y_val),
    ("Test", y_test),
]:
    print(
        f"{name:12s}: "
        f"total={len(y):,}, "
        f"buggy={y.sum():,}, "
        f"rate={y.mean():.4f}"
    )


# ============================================================
# CLASS WEIGHT
# ============================================================

train_negative = np.sum(y_train == 0)
train_positive = np.sum(y_train == 1)

scale_pos_weight = (
    train_negative /
    train_positive
)

print(
    f"\nscale_pos_weight = "
    f"{scale_pos_weight:.6f}"
)


# ============================================================
# MODELS
# ============================================================

models = {

    "RF": RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),

    "XGB": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    ),

    "LGBM": LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    ),
}


# ============================================================
# TRAIN BASE MODELS
# ============================================================

val_probabilities = {}
test_probabilities = {}

all_results = {
    "validation": {},
    "test": {},
}

for name, model in models.items():

    print("\n" + "=" * 80)
    print(f"TRAINING {name}")
    print("=" * 80)

    model.fit(
        X_train,
        y_train
    )

    val_prob = model.predict_proba(
        X_val
    )[:, 1]

    test_prob = model.predict_proba(
        X_test
    )[:, 1]

    val_probabilities[name] = val_prob
    test_probabilities[name] = test_prob

    val_result = metrics(
        y_val,
        val_prob
    )

    test_result = metrics(
        y_test,
        test_prob
    )

    all_results["validation"][name] = val_result
    all_results["test"][name] = test_result

    print_metrics(
        f"{name} - VALIDATION",
        val_result
    )

    print_metrics(
        f"{name} - TEST",
        test_result
    )

    # Save model
    import joblib

    joblib.dump(
        model,
        os.path.join(
            OUT_DIR,
            f"{name.lower()}_model.joblib"
        )
    )


# ============================================================
# VALIDATION-ONLY WEIGHT SEARCH
# ============================================================

print("\n" + "=" * 80)
print("VALIDATION-ONLY WEIGHT SEARCH")
print("=" * 80)

best = None

grid = np.arange(
    0.0,
    1.0001,
    0.05
)

for w_rf in grid:

    for w_xgb in grid:

        w_lgbm = 1.0 - w_rf - w_xgb

        if w_lgbm < -1e-9:
            continue

        blend_val = (
            w_rf * val_probabilities["RF"]
            + w_xgb * val_probabilities["XGB"]
            + w_lgbm * val_probabilities["LGBM"]
        )

        result = metrics(
            y_val,
            blend_val
        )

        score = (
            result["f1"],
            result["mcc"],
            result["roc_auc"],
            result["pr_auc"],
        )

        if best is None or score > best["score"]:

            best = {
                "weights": {
                    "RF": float(w_rf),
                    "XGB": float(w_xgb),
                    "LGBM": float(w_lgbm),
                },
                "score": score,
                "metrics": result,
            }


print("\nSelected validation weights:")

for model_name, weight in best["weights"].items():
    print(
        f"  {model_name}: {weight:.2f}"
    )

print_metrics(
    "BEST BLEND - VALIDATION",
    best["metrics"]
)


# ============================================================
# FINAL TEST BLEND
# ============================================================

w = best["weights"]

blend_test = (
    w["RF"] * test_probabilities["RF"]
    + w["XGB"] * test_probabilities["XGB"]
    + w["LGBM"] * test_probabilities["LGBM"]
)

blend_test_result = metrics(
    y_test,
    blend_test
)

all_results["test"]["BLEND"] = (
    blend_test_result
)

print_metrics(
    "FINAL BLEND - TEST",
    blend_test_result
)


# ============================================================
# ROC CURVE
# ============================================================

plt.figure(figsize=(8, 6))

for name in [
    "RF",
    "XGB",
    "LGBM",
]:

    fpr, tpr, _ = roc_curve(
        y_test,
        test_probabilities[name]
    )

    auc = roc_auc_score(
        y_test,
        test_probabilities[name]
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={auc:.4f})"
    )


fpr, tpr, _ = roc_curve(
    y_test,
    blend_test
)

auc = roc_auc_score(
    y_test,
    blend_test
)

plt.plot(
    fpr,
    tpr,
    label=f"Blend (AUC={auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(
    "Experiment 8 - Test ROC Curve"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "experiment_8_test_roc.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# PR CURVE
# ============================================================

plt.figure(figsize=(8, 6))

for name in [
    "RF",
    "XGB",
    "LGBM",
]:

    precision, recall, _ = precision_recall_curve(
        y_test,
        test_probabilities[name]
    )

    ap = average_precision_score(
        y_test,
        test_probabilities[name]
    )

    plt.plot(
        recall,
        precision,
        label=f"{name} (AP={ap:.4f})"
    )


precision, recall, _ = precision_recall_curve(
    y_test,
    blend_test
)

ap = average_precision_score(
    y_test,
    blend_test
)

plt.plot(
    recall,
    precision,
    label=f"Blend (AP={ap:.4f})"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(
    "Experiment 8 - Test Precision-Recall Curve"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "experiment_8_test_pr.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# SAVE RESULTS
# ============================================================

results_output = {
    "experiment": "Experiment 8 - JIT + CodeBERT + Reduced LLM",

    "feature_configuration": {
        "jit_features": len(JIT_FEATURES),
        "codebert_features": 384,
        "llm_features": len(LLM_REDUCED_FEATURES),
        "total_features": EXPECTED_TOTAL_FEATURES,
        "llm_feature_list": LLM_REDUCED_FEATURES,
    },

    "methodology": {
        "split": "project-wise chronological 70/15/15",
        "pca": "fit on train only",
        "llm_imputation": "median fitted on train only",
        "resampling": False,
        "random_split": False,
        "threshold": 0.5,
        "blend_selection": "validation only",
    },

    "validation_weights": best["weights"],

    "validation": all_results["validation"],
    "test": all_results["test"],
}


with open(
    os.path.join(
        OUT_DIR,
        "metrics.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results_output,
        f,
        indent=2
    )


# ============================================================
# SUMMARY CSV
# ============================================================

rows = []

for split in [
    "validation",
    "test",
]:

    for model_name, result in all_results[split].items():

        rows.append({
            "split": split,
            "model": model_name,
            "accuracy": result["accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "mcc": result["mcc"],
            "roc_auc": result["roc_auc"],
            "pr_auc": result["pr_auc"],
        })


summary_df = pd.DataFrame(rows)

summary_df.to_csv(
    os.path.join(
        OUT_DIR,
        "metrics_summary.csv"
    ),
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("EXPERIMENT 8 COMPLETE")
print("=" * 80)

print("\nFeature configuration:")
print("  JIT       = 12")
print("  CodeBERT  = 384")
print("  LLM       = 14")
print("  TOTAL     = 410")

print("\nSelected blend weights:")
print(best["weights"])

print("\nFINAL TEST BLEND:")
print_metrics(
    "Experiment 8",
    blend_test_result
)

print("\nOutputs:")
print(
    OUT_DIR
)

print(
    os.path.join(
        OUT_DIR,
        "metrics.json"
    )
)

print(
    os.path.join(
        OUT_DIR,
        "metrics_summary.csv"
    )
)

print(
    os.path.join(
        PLOTS_DIR,
        "experiment_8_test_roc.png"
    )
)

print(
    os.path.join(
        PLOTS_DIR,
        "experiment_8_test_pr.png"
    )
)