"""
EXPERIMENT 9
============

REDUNDANCY-AWARE LLM FEATURE SELECTION

Pipeline
--------
JIT (12)
    +
CodeBERT PCA (384)
    +
Selected LLM features
    =
Final model

LLM feature-selection methodology
---------------------------------
1. Load the 7 raw categorical LLM dimensions
   and 14 confidence/margin numerical features.

2. One-hot encode categorical features using TRAIN ONLY.

3. Median-impute numerical confidence/margin features using
   TRAIN ONLY.

4. Combine into the original 51-dimensional LLM representation.

5. Compute mutual information between each LLM feature and
   the buggy label using TRAIN ONLY.

6. Remove redundant LLM features using pairwise correlation
   calculated on TRAIN ONLY.

7. Rank the remaining features by mutual information.

8. Select the top K LLM features.

9. Build:
       12 JIT
       + 384 CodeBERT
       + K selected LLM

10. Train RF / XGBoost / LightGBM.

11. Select ensemble weights using VALIDATION ONLY.

12. Evaluate TEST once.

IMPORTANT
---------
No test data is used for:
    - OHE fitting
    - imputation
    - correlation filtering
    - mutual information
    - feature selection
    - blend-weight selection

This makes the feature-selection process leakage-safe.
"""

import os
import json
import ast
import warnings

import numpy as np
import pandas as pd

import matplotlib

# Prevent Windows Tkinter/backend problems
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
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
    roc_curve,
    precision_recall_curve,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import joblib

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = r"D:\Hybrid-Feature-Fusion-JIT-SDP"

JIT_DIR = os.path.join(
    ROOT,
    "results",
    "chronological_splits"
)

PCA_DIR = os.path.join(
    ROOT,
    "results",
    "pca"
)

LLM_DIR = os.path.join(
    ROOT,
    "data",
    "llm_experiment_dataset"
)

OUT_DIR = os.path.join(
    ROOT,
    "results",
    "experiment_9_redundancy_aware_llm_selection"
)

PLOTS_DIR = os.path.join(
    OUT_DIR,
    "plots"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)

os.makedirs(
    PLOTS_DIR,
    exist_ok=True
)


# ============================================================
# EXPERIMENT PARAMETERS
# ============================================================

# Number of LLM model-ready features to retain.
#
# 14 is intentionally chosen so Experiment 9 can be
# compared directly with Experiment 8.
#
# You can later change this to 5, 8, 10, etc.
TOP_K = 14

# Pairwise correlation threshold.
#
# If two LLM features have absolute Pearson correlation
# greater than this value, one of them is removed.
CORRELATION_THRESHOLD = 0.90

RANDOM_STATE = 42


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


LLM_CATEGORICAL_FEATURES = [
    "intent",
    "change",
    "risk",
    "complexity",
    "scope",
    "test",
    "security",
]


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
ID_COL = "commit_id"


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def normalize_target(series):
    """
    Convert buggy labels into integer 0/1.
    """

    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(
            series,
            errors="coerce"
        ).astype(int)

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

    result = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    if result.isna().any():
        bad = series[result.isna()].unique()

        raise ValueError(
            f"Unknown target values: {bad}"
        )

    return result.astype(int)


def parse_embedding(value):
    """
    Parse 384-dimensional PCA embedding.
    """

    if isinstance(
        value,
        (list, tuple, np.ndarray)
    ):
        return np.asarray(
            value,
            dtype=np.float32
        )

    if pd.isna(value):
        return None

    text = str(value).strip()

    try:

        return np.asarray(
            json.loads(text),
            dtype=np.float32
        )

    except Exception:

        try:

            return np.asarray(
                ast.literal_eval(text),
                dtype=np.float32
            )

        except Exception:

            raise ValueError(
                f"Could not parse embedding: "
                f"{text[:100]}"
            )


def load_pca_file(path):

    df = pd.read_csv(path)

    embeddings = np.vstack(
        df["embedding"]
        .apply(parse_embedding)
        .values
    )

    if embeddings.shape[1] != 384:

        raise ValueError(
            f"{path}: expected 384 dimensions, "
            f"got {embeddings.shape[1]}"
        )

    return df, embeddings


def load_llm_dataset():

    csv_files = [
        os.path.join(
            LLM_DIR,
            f
        )
        for f in os.listdir(LLM_DIR)
        if f.lower().endswith(".csv")
    ]

    if not csv_files:

        raise FileNotFoundError(
            f"No CSV files found in {LLM_DIR}"
        )

    frames = []

    for path in sorted(csv_files):

        print(
            f"Loading LLM file: {path}"
        )

        frames.append(
            pd.read_csv(path)
        )

    llm = pd.concat(
        frames,
        ignore_index=True
    )

    if ID_COL not in llm.columns:

        raise ValueError(
            "LLM dataset does not contain commit_id"
        )

    if llm[ID_COL].duplicated().any():

        duplicate_count = (
            llm[ID_COL]
            .duplicated()
            .sum()
        )

        print(
            f"WARNING: {duplicate_count} "
            f"duplicate LLM IDs found."
        )

        llm = llm.drop_duplicates(
            subset=[ID_COL],
            keep="first"
        )

    required = (
        LLM_CATEGORICAL_FEATURES
        +
        LLM_NUMERICAL_FEATURES
    )

    missing = [
        c for c in required
        if c not in llm.columns
    ]

    if missing:

        raise ValueError(
            "Missing LLM columns:\n"
            + "\n".join(missing)
        )

    return llm[
        [ID_COL] + required
    ].copy()


def get_jit_matrix(df):

    missing = [
        c
        for c in JIT_FEATURES
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing JIT features: {missing}"
        )

    return (
        df[JIT_FEATURES]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .to_numpy(
            dtype=np.float32
        )
    )


def calculate_metrics(
    y_true,
    probabilities
):

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        predictions
    )

    return {

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

        "confusion_matrix":
            cm.tolist(),
    }


def print_metrics(
    name,
    result
):

    print(
        f"\n{name}"
    )

    print(
        "-" * 70
    )

    print(
        f"Accuracy : "
        f"{result['accuracy']:.6f}"
    )

    print(
        f"Precision: "
        f"{result['precision']:.6f}"
    )

    print(
        f"Recall   : "
        f"{result['recall']:.6f}"
    )

    print(
        f"F1       : "
        f"{result['f1']:.6f}"
    )

    print(
        f"MCC      : "
        f"{result['mcc']:.6f}"
    )

    print(
        f"ROC-AUC  : "
        f"{result['roc_auc']:.6f}"
    )

    print(
        f"PR-AUC   : "
        f"{result['pr_auc']:.6f}"
    )

    print(
        "CM       :",
        np.array(
            result["confusion_matrix"]
        )
    )


# ============================================================
# LOAD CHRONOLOGICAL JIT SPLITS
# ============================================================

print(
    "=" * 80
)

print(
    "EXPERIMENT 9: "
    "REDUNDANCY-AWARE LLM FEATURE SELECTION"
)

print(
    "=" * 80
)

print(
    "\nConfiguration:"
)

print(
    f"  JIT features       : "
    f"{len(JIT_FEATURES)}"
)

print(
    "  CodeBERT features  : 384"
)

print(
    f"  LLM raw categorical: "
    f"{len(LLM_CATEGORICAL_FEATURES)}"
)

print(
    f"  LLM numerical      : "
    f"{len(LLM_NUMERICAL_FEATURES)}"
)

print(
    f"  LLM model features : "
    f"51 expected"
)

print(
    f"  Correlation cutoff : "
    f"{CORRELATION_THRESHOLD}"
)

print(
    f"  Selected LLM K     : "
    f"{TOP_K}"
)


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
    f"\nTrain: "
    f"{len(train_jit):,}"
)

print(
    f"Validation: "
    f"{len(val_jit):,}"
)

print(
    f"Test: "
    f"{len(test_jit):,}"
)


# ============================================================
# LOAD PCA CODEBERT
# ============================================================

print(
    "\nLoading PCA CodeBERT..."
)

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
    "CodeBERT:",
    train_cb.shape,
    val_cb.shape,
    test_cb.shape
)


# ============================================================
# LOAD LLM
# ============================================================

print(
    "\nLoading LLM dataset..."
)

llm = load_llm_dataset()

print(
    f"LLM rows: "
    f"{len(llm):,}"
)


# ============================================================
# ALIGN LLM WITH EACH SPLIT
# ============================================================

llm_indexed = (
    llm
    .set_index(ID_COL)
)


def align_llm(
    jit_df,
    pca_df,
    split_name
):

    jit_ids = (
        jit_df[ID_COL]
        .astype(str)
    )

    pca_ids = (
        pca_df[ID_COL]
        .astype(str)
    )

    if not np.array_equal(
        jit_ids.values,
        pca_ids.values
    ):

        raise ValueError(
            f"{split_name}: "
            "JIT/PCA commit IDs do not match."
        )

    if jit_ids.duplicated().any():

        raise ValueError(
            f"{split_name}: "
            "duplicate commit IDs."
        )

    aligned = (
        llm_indexed
        .reindex(jit_ids)
        .reset_index(drop=True)
    )

    if aligned.isna().all(axis=None):

        raise ValueError(
            f"{split_name}: "
            "LLM alignment failed."
        )

    missing = (
        aligned.isna()
        .any(axis=1)
        .sum()
    )

    if missing > 0:

        raise ValueError(
            f"{split_name}: "
            f"{missing} missing LLM rows."
        )

    y = normalize_target(
        jit_df[TARGET]
    ).to_numpy()

    print(
        f"{split_name}: "
        f"rows={len(jit_df):,}, "
        f"buggy={y.sum():,}"
    )

    return aligned, y


train_llm, y_train = align_llm(
    train_jit,
    train_pca_df,
    "TRAIN"
)

val_llm, y_val = align_llm(
    val_jit,
    val_pca_df,
    "VALIDATION"
)

test_llm, y_test = align_llm(
    test_jit,
    test_pca_df,
    "TEST"
)


# ============================================================
# TRAIN-ONLY LLM PREPROCESSING
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "LLM PREPROCESSING"
)

print(
    "=" * 80
)


# ------------------------------------------------------------
# CATEGORICAL
# ------------------------------------------------------------

print(
    "\nFitting OneHotEncoder on TRAIN only..."
)

try:

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

except TypeError:

    # Compatibility with older sklearn
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False
    )


train_cat = encoder.fit_transform(
    train_llm[
        LLM_CATEGORICAL_FEATURES
    ].astype(str)
)

val_cat = encoder.transform(
    val_llm[
        LLM_CATEGORICAL_FEATURES
    ].astype(str)
)

test_cat = encoder.transform(
    test_llm[
        LLM_CATEGORICAL_FEATURES
    ].astype(str)
)


cat_feature_names = (
    encoder
    .get_feature_names_out(
        LLM_CATEGORICAL_FEATURES
    )
)


print(
    f"Categorical one-hot features: "
    f"{train_cat.shape[1]}"
)


# ------------------------------------------------------------
# NUMERICAL
# ------------------------------------------------------------

print(
    "\nFitting numerical imputer on TRAIN only..."
)

imputer = SimpleImputer(
    strategy="median"
)

train_num = imputer.fit_transform(
    train_llm[
        LLM_NUMERICAL_FEATURES
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )
)

val_num = imputer.transform(
    val_llm[
        LLM_NUMERICAL_FEATURES
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )
)

test_num = imputer.transform(
    test_llm[
        LLM_NUMERICAL_FEATURES
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )
)


# ============================================================
# COMBINE INTO ORIGINAL 51-D LLM REPRESENTATION
# ============================================================

train_llm_full = np.hstack([
    train_cat,
    train_num,
]).astype(
    np.float32
)

val_llm_full = np.hstack([
    val_cat,
    val_num,
]).astype(
    np.float32
)

test_llm_full = np.hstack([
    test_cat,
    test_num,
]).astype(
    np.float32
)


llm_feature_names = (
    list(cat_feature_names)
    +
    LLM_NUMERICAL_FEATURES
)


print(
    "\nFull LLM representation:"
)

print(
    f"Train: "
    f"{train_llm_full.shape}"
)

print(
    f"Validation: "
    f"{val_llm_full.shape}"
)

print(
    f"Test: "
    f"{test_llm_full.shape}"
)

if train_llm_full.shape[1] != 51:

    raise ValueError(
        "Expected 51 model-ready LLM features, "
        f"found {train_llm_full.shape[1]}"
    )


# ============================================================
# MUTUAL INFORMATION
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "TRAIN-ONLY MUTUAL INFORMATION"
)

print(
    "=" * 80
)

print(
    "\nComputing MI on TRAIN only..."
)

mi_scores = mutual_info_classif(
    train_llm_full,
    y_train,
    random_state=RANDOM_STATE
)


mi_df = pd.DataFrame({
    "feature": llm_feature_names,
    "mutual_information": mi_scores,
})


mi_df = mi_df.sort_values(
    "mutual_information",
    ascending=False
).reset_index(
    drop=True
)

mi_df[
    "mi_rank"
] = np.arange(
    1,
    len(mi_df) + 1
)


print(
    "\nTop 20 features by MI:"
)

print(
    mi_df.head(20).to_string(
        index=False
    )
)


mi_df.to_csv(
    os.path.join(
        OUT_DIR,
        "train_only_mutual_information.csv"
    ),
    index=False
)


# ============================================================
# REDUNDANCY FILTER
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "TRAIN-ONLY REDUNDANCY FILTER"
)

print(
    "=" * 80
)

print(
    f"\nCorrelation threshold: "
    f"|r| >= {CORRELATION_THRESHOLD}"
)


# Compute Pearson correlation on TRAIN only.
corr = np.corrcoef(
    train_llm_full,
    rowvar=False
)

corr = np.nan_to_num(
    corr,
    nan=0.0
)


# ------------------------------------------------------------
# Greedy MI-priority redundancy filtering
# ------------------------------------------------------------
#
# Important:
# Features are considered in descending MI order.
#
# A feature is retained if it is not highly correlated
# with an already retained feature.
#
# This means:
#
#     higher-MI feature wins
#     lower-MI redundant feature is removed
#
# This is entirely train-only.
# ------------------------------------------------------------

selected_indices = []
removed_features = []

for idx in mi_df.index:

    original_index = (
        llm_feature_names.index(
            mi_df.loc[
                idx,
                "feature"
            ]
        )
    )

    feature_name = (
        mi_df.loc[
            idx,
            "feature"
        ]
    )

    if len(selected_indices) == 0:

        selected_indices.append(
            original_index
        )

        continue

    correlations = np.abs(
        corr[
            original_index,
            selected_indices
        ]
    )

    max_corr = (
        np.max(correlations)
    )

    if max_corr >= CORRELATION_THRESHOLD:

        matched_index = (
            selected_indices[
                np.argmax(correlations)
            ]
        )

        removed_features.append({
            "feature": feature_name,
            "reason": "high_correlation",
            "correlated_with":
                llm_feature_names[
                    matched_index
                ],
            "absolute_correlation":
                float(max_corr),
        })

    else:

        selected_indices.append(
            original_index
        )


redundancy_kept = [
    llm_feature_names[i]
    for i in selected_indices
]


print(
    f"\nFeatures before redundancy filtering: "
    f"{len(llm_feature_names)}"
)

print(
    f"Features after redundancy filtering: "
    f"{len(redundancy_kept)}"
)

print(
    f"Features removed as redundant: "
    f"{len(removed_features)}"
)


# ============================================================
# SELECT TOP K BY MI FROM NON-REDUNDANT FEATURES
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "FINAL REDUNDANCY-AWARE FEATURE SELECTION"
)

print(
    "=" * 80
)


# Keep only non-redundant features
non_redundant_mi = mi_df[
    mi_df["feature"].isin(
        redundancy_kept
    )
].copy()


non_redundant_mi = (
    non_redundant_mi
    .sort_values(
        "mutual_information",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


if len(non_redundant_mi) < TOP_K:

    raise ValueError(
        f"Only {len(non_redundant_mi)} "
        f"non-redundant features remain, "
        f"but TOP_K={TOP_K}."
    )


selected_feature_names = (
    non_redundant_mi
    .head(TOP_K)
    ["feature"]
    .tolist()
)


selected_feature_indices = [
    llm_feature_names.index(
        name
    )
    for name in selected_feature_names
]


print(
    f"\nSelected {TOP_K} LLM features:"
)

for rank, name in enumerate(
    selected_feature_names,
    start=1
):

    mi_value = float(
        mi_df.loc[
            mi_df["feature"] == name,
            "mutual_information"
        ].iloc[0]
    )

    print(
        f"{rank:2d}. "
        f"{name:<40} "
        f"MI={mi_value:.6f}"
    )


# ============================================================
# SAVE FEATURE-SELECTION METADATA
# ============================================================

selection_metadata = {

    "method":
        "MI ranking followed by greedy "
        "correlation redundancy filtering",

    "correlation_threshold":
        CORRELATION_THRESHOLD,

    "top_k":
        TOP_K,

    "original_llm_features":
        len(llm_feature_names),

    "non_redundant_features":
        len(redundancy_kept),

    "selected_features":
        selected_feature_names,

    "removed_redundant_features":
        removed_features,

    "leakage_control":
        "All preprocessing, MI and correlation "
        "selection fitted/calculated on TRAIN only.",
}


with open(
    os.path.join(
        OUT_DIR,
        "feature_selection_metadata.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        selection_metadata,
        f,
        indent=2
    )


# Save selected feature table
selected_table = (
    non_redundant_mi
    .head(TOP_K)
    .copy()
)

selected_table[
    "selected_rank"
] = np.arange(
    1,
    TOP_K + 1
)

selected_table.to_csv(
    os.path.join(
        OUT_DIR,
        "selected_llm_features.csv"
    ),
    index=False
)


# ============================================================
# EXTRACT SELECTED LLM FEATURES
# ============================================================

X_train_llm = (
    train_llm_full[
        :,
        selected_feature_indices
    ]
)

X_val_llm = (
    val_llm_full[
        :,
        selected_feature_indices
    ]
)

X_test_llm = (
    test_llm_full[
        :,
        selected_feature_indices
    ]
)


print(
    "\nSelected LLM matrices:"
)

print(
    f"Train: "
    f"{X_train_llm.shape}"
)

print(
    f"Validation: "
    f"{X_val_llm.shape}"
)

print(
    f"Test: "
    f"{X_test_llm.shape}"
)


# ============================================================
# BUILD JIT + CODEBERT + SELECTED LLM
# ============================================================

X_train_jit = get_jit_matrix(
    train_jit
)

X_val_jit = get_jit_matrix(
    val_jit
)

X_test_jit = get_jit_matrix(
    test_jit
)


X_train = np.hstack([
    X_train_jit,
    train_cb,
    X_train_llm,
]).astype(
    np.float32
)

X_val = np.hstack([
    X_val_jit,
    val_cb,
    X_val_llm,
]).astype(
    np.float32
)

X_test = np.hstack([
    X_test_jit,
    test_cb,
    X_test_llm,
]).astype(
    np.float32
)


EXPECTED_TOTAL = (
    len(JIT_FEATURES)
    + 384
    + TOP_K
)


print(
    "\nFinal feature matrices:"
)

print(
    f"Train: "
    f"{X_train.shape}"
)

print(
    f"Validation: "
    f"{X_val.shape}"
)

print(
    f"Test: "
    f"{X_test.shape}"
)


assert (
    X_train.shape[1]
    == EXPECTED_TOTAL
)

assert (
    X_val.shape[1]
    == EXPECTED_TOTAL
)

assert (
    X_test.shape[1]
    == EXPECTED_TOTAL
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print(
    "\nLabel distribution:"
)

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

train_negative = np.sum(
    y_train == 0
)

train_positive = np.sum(
    y_train == 1
)

scale_pos_weight = (
    train_negative
    /
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

    "RF":
        RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),

    "XGB":
        XGBClassifier(
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

    "LGBM":
        LGBMClassifier(
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
# TRAIN MODELS
# ============================================================

validation_probabilities = {}
test_probabilities = {}

all_results = {
    "validation": {},
    "test": {},
}


for name, model in models.items():

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"TRAINING {name}"
    )

    print(
        "=" * 80
    )

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

    validation_probabilities[
        name
    ] = val_prob

    test_probabilities[
        name
    ] = test_prob

    val_result = calculate_metrics(
        y_val,
        val_prob
    )

    test_result = calculate_metrics(
        y_test,
        test_prob
    )

    all_results[
        "validation"
    ][name] = val_result

    all_results[
        "test"
    ][name] = test_result

    print_metrics(
        f"{name} - VALIDATION",
        val_result
    )

    print_metrics(
        f"{name} - TEST",
        test_result
    )

    joblib.dump(
        model,
        os.path.join(
            OUT_DIR,
            f"{name.lower()}_model.joblib"
        )
    )


# ============================================================
# VALIDATION-ONLY ENSEMBLE WEIGHT SEARCH
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "VALIDATION-ONLY WEIGHT SEARCH"
)

print(
    "=" * 80
)


best = None

grid = np.arange(
    0.0,
    1.0001,
    0.05
)


for w_rf in grid:

    for w_xgb in grid:

        w_lgbm = (
            1.0
            - w_rf
            - w_xgb
        )

        if w_lgbm < -1e-9:

            continue

        blend_val = (
            w_rf
            * validation_probabilities[
                "RF"
            ]

            +

            w_xgb
            * validation_probabilities[
                "XGB"
            ]

            +

            w_lgbm
            * validation_probabilities[
                "LGBM"
            ]
        )

        result = calculate_metrics(
            y_val,
            blend_val
        )

        score = (
            result["f1"],
            result["mcc"],
            result["roc_auc"],
            result["pr_auc"],
        )

        if (
            best is None
            or score > best["score"]
        ):

            best = {

                "weights": {
                    "RF":
                        float(w_rf),

                    "XGB":
                        float(w_xgb),

                    "LGBM":
                        float(w_lgbm),
                },

                "score":
                    score,

                "metrics":
                    result,
            }


print(
    "\nSelected validation weights:"
)

for name, weight in (
    best["weights"].items()
):

    print(
        f"  {name}: "
        f"{weight:.2f}"
    )


print_metrics(
    "BEST BLEND - VALIDATION",
    best["metrics"]
)


# ============================================================
# FINAL TEST BLEND
# ============================================================

weights = best[
    "weights"
]


blend_test = (

    weights["RF"]
    * test_probabilities["RF"]

    +

    weights["XGB"]
    * test_probabilities["XGB"]

    +

    weights["LGBM"]
    * test_probabilities["LGBM"]
)


blend_test_result = calculate_metrics(
    y_test,
    blend_test
)


all_results[
    "test"
]["BLEND"] = (
    blend_test_result
)


print_metrics(
    "FINAL BLEND - TEST",
    blend_test_result
)


# ============================================================
# ROC PLOT
# ============================================================

plt.figure(
    figsize=(8, 6)
)


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
        label=(
            f"{name} "
            f"(AUC={auc:.4f})"
        )
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
    label=(
        f"Blend "
        f"(AUC={auc:.4f})"
    )
)


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Experiment 9 - Test ROC Curve"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "experiment_9_test_roc.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# PR PLOT
# ============================================================

plt.figure(
    figsize=(8, 6)
)


for name in [
    "RF",
    "XGB",
    "LGBM",
]:

    precision, recall, _ = (
        precision_recall_curve(
            y_test,
            test_probabilities[name]
        )
    )

    ap = average_precision_score(
        y_test,
        test_probabilities[name]
    )

    plt.plot(
        recall,
        precision,
        label=(
            f"{name} "
            f"(AP={ap:.4f})"
        )
    )


precision, recall, _ = (
    precision_recall_curve(
        y_test,
        blend_test
    )
)

ap = average_precision_score(
    y_test,
    blend_test
)


plt.plot(
    recall,
    precision,
    label=(
        f"Blend "
        f"(AP={ap:.4f})"
    )
)


plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Experiment 9 - Test Precision-Recall Curve"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "experiment_9_test_pr.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# SAVE RESULTS
# ============================================================

results_output = {

    "experiment":
        "Experiment 9 - "
        "Redundancy-Aware LLM Feature Selection",

    "feature_configuration": {

        "jit_features":
            len(JIT_FEATURES),

        "codebert_features":
            384,

        "original_llm_features":
            len(llm_feature_names),

        "selected_llm_features":
            TOP_K,

        "total_model_features":
            EXPECTED_TOTAL,

        "selected_llm_feature_names":
            selected_feature_names,
    },

    "selection_method": {

        "mutual_information":
            "TRAIN only",

        "correlation":
            "TRAIN only",

        "correlation_threshold":
            CORRELATION_THRESHOLD,

        "selection":
            "MI-priority greedy redundancy filtering",

    },

    "methodology": {

        "split":
            "project-wise chronological 70/15/15",

        "pca":
            "fit on train only",

        "one_hot_encoding":
            "fit on train only",

        "median_imputation":
            "fit on train only",

        "resampling":
            False,

        "random_split":
            False,

        "threshold":
            0.5,

        "blend_selection":
            "validation only",
    },

    "validation_weights":
        best["weights"],

    "validation":
        all_results["validation"],

    "test":
        all_results["test"],
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

summary_rows = []


for split in [
    "validation",
    "test",
]:

    for model_name, result in (
        all_results[split].items()
    ):

        summary_rows.append({

            "split":
                split,

            "model":
                model_name,

            "accuracy":
                result["accuracy"],

            "precision":
                result["precision"],

            "recall":
                result["recall"],

            "f1":
                result["f1"],

            "mcc":
                result["mcc"],

            "roc_auc":
                result["roc_auc"],

            "pr_auc":
                result["pr_auc"],
        })


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    os.path.join(
        OUT_DIR,
        "metrics_summary.csv"
    ),
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "EXPERIMENT 9 COMPLETE"
)

print(
    "=" * 80
)

print(
    "\nFeature configuration:"
)

print(
    f"  JIT      = "
    f"{len(JIT_FEATURES)}"
)

print(
    "  CodeBERT = 384"
)

print(
    f"  LLM      = "
    f"{TOP_K}"
)

print(
    f"  TOTAL    = "
    f"{EXPECTED_TOTAL}"
)

print(
    "\nSelected LLM features:"
)

for i, feature in enumerate(
    selected_feature_names,
    start=1
):

    print(
        f"  {i:2d}. {feature}"
    )


print(
    "\nSelected blend weights:"
)

print(
    best["weights"]
)


print(
    "\nFINAL TEST BLEND:"
)

print_metrics(
    "Experiment 9",
    blend_test_result
)


print(
    "\nOutputs:"
)

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
        OUT_DIR,
        "feature_selection_metadata.json"
    )
)

print(
    os.path.join(
        OUT_DIR,
        "selected_llm_features.csv"
    )
)
print(
    os.path.join(
        PLOTS_DIR,
        "experiment_9_test_roc.png"
    )
)

print(
    os.path.join(
        PLOTS_DIR,
        "experiment_9_test_pr.png"
    )
)