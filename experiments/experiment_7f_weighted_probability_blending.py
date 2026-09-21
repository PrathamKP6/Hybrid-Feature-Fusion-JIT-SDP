"""
EXPERIMENT 7F: WEIGHTED PROBABILITY BLENDING
============================================================

Purpose
-------
Evaluate weighted probability blending of three base learners on the
FULL three-way feature fusion:

    12 JIT + 384 CodeBERT PCA + 51 LLM = 447 features

Base models:
    1. Random Forest
    2. XGBoost
    3. LightGBM

Methodology
-----------
- Project-wise chronological 70/15/15 split (precomputed)
- PCA was fitted on TRAIN only (384 CodeBERT components)
- LLM one-hot encoding fitted on TRAIN only
- LLM numerical median imputation fitted on TRAIN only
- No random split
- No resampling
- Base models trained on TRAIN only
- Validation probabilities used ONLY to select blend weights
- TEST remains untouched until final evaluation
- Direct classification threshold = 0.5
- Blend weights sum to 1.0
- Weight grid step = 0.05
- Selection criterion:
      validation F1
      -> MCC
      -> ROC-AUC
      -> PR-AUC
  (all calculated on validation only)
- Final ROC/PR plots are generated for the weighted blend

Output
------
results/experiment_7f_probability_blending/
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PCA_DIR = ROOT / "results" / "pca"
LLM_DIR = ROOT / "data" / "llm_experiment_dataset"

TRAIN_PCA = PCA_DIR / "train_pca384.csv"
VAL_PCA = PCA_DIR / "validation_pca384.csv"
TEST_PCA = PCA_DIR / "test_pca384.csv"

TRAIN_LLM = LLM_DIR / "train.csv"
VAL_LLM = LLM_DIR / "validation.csv"
TEST_LLM = LLM_DIR / "test.csv"

OUTPUT_DIR = ROOT / "results" / "experiment_7f_probability_blending"
MODELS_DIR = OUTPUT_DIR / "models"
PLOTS_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXPERIMENT CONSTANTS
# ============================================================

JIT_FEATURES = [
    "la", "ld", "nf", "ns", "nd", "ent",
    "ndev", "age", "nuc", "aexp", "arexp", "asexp"
]

TARGET = "buggy"
ID_COL = "commit_id"

CODEBERT_SOURCE_DIM = 384
CODEBERT_DIM = 384
LLM_DIM = 51
JIT_DIM = 12
TOTAL_FEATURES = 447

LLM_CATEGORICAL = [
    "intent",
    "change",
    "risk",
    "complexity",
    "scope",
    "test",
    "security",
]

LLM_NUMERICAL = [
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

RF_PARAMS = dict(
    n_estimators=300,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

XGB_BASE_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

LGBM_BASE_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)

BLEND_STEP = 0.05
THRESHOLD = 0.50


# ============================================================
# HELPERS
# ============================================================

def normalize_target(series):
    """Convert common buggy-label representations to {0,1}."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce")
        if vals.isna().any():
            raise ValueError("Target contains non-numeric values after numeric conversion.")
        unique = set(vals.unique().tolist())
        if not unique.issubset({0, 1}):
            raise ValueError(f"Unexpected numeric target values: {sorted(unique)}")
        return vals.astype(int)

    mapping = {
        "0": 0,
        "1": 1,
        "false": 0,
        "true": 1,
        "non-buggy": 0,
        "non_buggy": 0,
        "nonbuggy": 0,
        "buggy": 1,
        "bug": 1,
    }

    normalized = series.astype(str).str.strip().str.lower().map(mapping)

    if normalized.isna().any():
        bad = series[normalized.isna()].astype(str).unique()[:10]
        raise ValueError(f"Unknown target labels: {bad}")

    return normalized.astype(int)


def parse_embedding(value, expected_dim=384):
    """Parse a PCA embedding stored as JSON/list/string."""
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=np.float32)
    else:
        text = str(value).strip()
        try:
            arr = np.asarray(json.loads(text), dtype=np.float32)
        except Exception:
            text = text.strip("[]")
            arr = np.fromstring(text, sep=",", dtype=np.float32)

    arr = arr.reshape(-1)

    if len(arr) != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected_dim}, got {len(arr)}"
        )

    return arr


def extract_codebert(df, expected_dim=384):
    """Extract 384 PCA CodeBERT features from the embedding column."""
    if "embedding" not in df.columns:
        raise KeyError("Expected 'embedding' column in PCA dataset.")

    matrix = np.vstack(
        [parse_embedding(v, expected_dim) for v in df["embedding"].values]
    ).astype(np.float32)

    return matrix


def make_llm_preprocessor():
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )
    imputer = SimpleImputer(strategy="median")
    return encoder, imputer


def fit_transform_llm(train_df, val_df, test_df):
    """Fit LLM preprocessing on TRAIN only and transform all splits."""
    missing_cat = [c for c in LLM_CATEGORICAL if c not in train_df.columns]
    missing_num = [c for c in LLM_NUMERICAL if c not in train_df.columns]

    if missing_cat or missing_num:
        raise KeyError(
            f"Missing LLM columns. categorical={missing_cat}, numerical={missing_num}"
        )

    encoder, imputer = make_llm_preprocessor()

    X_train_cat = encoder.fit_transform(
        train_df[LLM_CATEGORICAL].fillna("UNKNOWN").astype(str)
    )
    X_val_cat = encoder.transform(
        val_df[LLM_CATEGORICAL].fillna("UNKNOWN").astype(str)
    )
    X_test_cat = encoder.transform(
        test_df[LLM_CATEGORICAL].fillna("UNKNOWN").astype(str)
    )

    X_train_num = imputer.fit_transform(train_df[LLM_NUMERICAL])
    X_val_num = imputer.transform(val_df[LLM_NUMERICAL])
    X_test_num = imputer.transform(test_df[LLM_NUMERICAL])

    X_train_llm = np.hstack([X_train_cat, X_train_num]).astype(np.float32)
    X_val_llm = np.hstack([X_val_cat, X_val_num]).astype(np.float32)
    X_test_llm = np.hstack([X_test_cat, X_test_num]).astype(np.float32)

    print(f"  LLM one-hot features: {X_train_cat.shape[1]}")
    print(f"  LLM numerical features: {X_train_num.shape[1]}")
    print(f"  LLM total features: {X_train_llm.shape[1]}")

    if X_train_llm.shape[1] != LLM_DIM:
        raise AssertionError(
            f"Expected {LLM_DIM} LLM features, got {X_train_llm.shape[1]}"
        )

    return X_train_llm, X_val_llm, X_test_llm, encoder, imputer


def build_feature_matrix(pca_df, llm_matrix):
    """Build JIT + full CodeBERT + LLM feature matrix."""
    missing_jit = [c for c in JIT_FEATURES if c not in pca_df.columns]
    if missing_jit:
        raise KeyError(f"Missing JIT columns: {missing_jit}")

    jit = pca_df[JIT_FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float32
    )

    codebert = extract_codebert(pca_df, CODEBERT_SOURCE_DIM)

    if codebert.shape[1] != CODEBERT_DIM:
        raise AssertionError(
            f"Expected {CODEBERT_DIM} CodeBERT features, got {codebert.shape[1]}"
        )

    if len(pca_df) != len(llm_matrix):
        raise AssertionError(
            f"Row mismatch: PCA={len(pca_df)}, LLM={len(llm_matrix)}"
        )

    X = np.hstack([jit, codebert, llm_matrix]).astype(np.float32)

    if X.shape[1] != TOTAL_FEATURES:
        raise AssertionError(
            f"Expected {TOTAL_FEATURES} fused features, got {X.shape[1]}"
        )

    return X


def validate_split_ids(train_pca, val_pca, test_pca, train_llm, val_llm, test_llm):
    """Validate uniqueness, cross-split separation and PCA/LLM alignment."""
    for name, df in [
        ("train PCA", train_pca),
        ("validation PCA", val_pca),
        ("test PCA", test_pca),
        ("train LLM", train_llm),
        ("validation LLM", val_llm),
        ("test LLM", test_llm),
    ]:
        if ID_COL not in df.columns:
            raise KeyError(f"{ID_COL} missing from {name}.")

        if df[ID_COL].duplicated().any():
            raise AssertionError(f"Duplicate commit IDs found in {name}.")

    pairs = [
        ("train", train_pca, train_llm),
        ("validation", val_pca, val_llm),
        ("test", test_pca, test_llm),
    ]

    for split, pca_df, llm_df in pairs:
        pca_ids = pca_df[ID_COL].astype(str).to_numpy()
        llm_ids = llm_df[ID_COL].astype(str).to_numpy()

        if not np.array_equal(pca_ids, llm_ids):
            raise AssertionError(f"{split}: PCA/LLM commit ID alignment failed.")

    train_ids = set(train_pca[ID_COL].astype(str))
    val_ids = set(val_pca[ID_COL].astype(str))
    test_ids = set(test_pca[ID_COL].astype(str))

    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise AssertionError("Cross-split commit ID overlap detected.")

    print("  [OK] Commit IDs unique and aligned.")
    print("  [OK] Cross-split commit ID overlap = 0.")


def validate_labels(pca_df, llm_df, split_name):
    y_pca = normalize_target(pca_df[TARGET]).to_numpy()
    y_llm = normalize_target(llm_df[TARGET]).to_numpy()

    if not np.array_equal(y_pca, y_llm):
        raise AssertionError(f"{split_name}: PCA/LLM labels do not match.")

    return y_pca


def metrics_from_probabilities(y_true, probabilities, threshold=0.5):
    pred = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred)

    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": cm.tolist(),
    }


def print_metrics(name, metrics):
    print(f"\n{name}")
    print(f"  Accuracy : {metrics['accuracy']:.6f}")
    print(f"  Precision: {metrics['precision']:.6f}")
    print(f"  Recall   : {metrics['recall']:.6f}")
    print(f"  F1       : {metrics['f1']:.6f}")
    print(f"  MCC      : {metrics['mcc']:.6f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.6f}")
    print(f"  PR-AUC   : {metrics['pr_auc']:.6f}")
    print(f"  CM       : {metrics['confusion_matrix']}")


def plot_roc_pr(y_true, probabilities, split_name):
    """Generate ROC and PR plots for the final weighted blend."""
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    precision, recall, _ = precision_recall_curve(y_true, probabilities)

    roc_auc = roc_auc_score(y_true, probabilities)
    pr_auc = average_precision_score(y_true, probabilities)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"Weighted Blend (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Experiment 7F - Weighted Probability Blend ROC ({split_name})")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"weighted_blend_{split_name.lower()}_roc.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label=f"Weighted Blend (AP = {pr_auc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Experiment 7F - Weighted Probability Blend PR ({split_name})")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"weighted_blend_{split_name.lower()}_pr.png", dpi=300)
    plt.close()


def generate_weight_grid(step=0.05):
    """
    Generate all non-negative RF/XGB/LGBM weights that sum to 1.

    Example:
        (1.00, 0.00, 0.00)
        (0.95, 0.05, 0.00)
        ...
        (0.00, 0.00, 1.00)
    """
    n = int(round(1.0 / step))
    weights = []

    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            w_rf = i / n
            w_xgb = j / n
            w_lgbm = k / n
            weights.append((w_rf, w_xgb, w_lgbm))

    return weights


def select_blend_weights(y_val, p_rf, p_xgb, p_lgbm):
    """
    Select blend weights using VALIDATION ONLY.

    Primary criterion:
        F1

    Tie-breakers:
        MCC
        ROC-AUC
        PR-AUC
    """
    candidates = []

    for w_rf, w_xgb, w_lgbm in generate_weight_grid(BLEND_STEP):
        p_blend = (
            w_rf * p_rf
            + w_xgb * p_xgb
            + w_lgbm * p_lgbm
        )

        m = metrics_from_probabilities(y_val, p_blend, THRESHOLD)

        candidates.append({
            "rf_weight": w_rf,
            "xgb_weight": w_xgb,
            "lightgbm_weight": w_lgbm,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "mcc": m["mcc"],
            "roc_auc": m["roc_auc"],
            "pr_auc": m["pr_auc"],
        })

    # Deterministic lexicographic selection.
    candidates.sort(
        key=lambda r: (
            r["f1"],
            r["mcc"],
            r["roc_auc"],
            r["pr_auc"],
        ),
        reverse=True,
    )

    best = candidates[0]
    return best, candidates


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 78)
    print("EXPERIMENT 7F: WEIGHTED PROBABILITY BLENDING")
    print("=" * 78)

    print("\nMethodology:")
    print("  Split: Project-wise chronological 70/15/15")
    print("  Features: 12 JIT + 384 CodeBERT + 51 LLM = 447")
    print("  Base models: Random Forest + XGBoost + LightGBM")
    print("  Blend: Weighted average of validation/test probabilities")
    print("  Weight grid step:", BLEND_STEP)
    print("  Weight selection: VALIDATION ONLY")
    print("  Selection: F1 -> MCC -> ROC-AUC -> PR-AUC")
    print("  PCA: fitted on TRAIN only")
    print("  LLM one-hot encoding: fitted on TRAIN only")
    print("  LLM imputation: fitted on TRAIN only")
    print("  Resampling: NONE")
    print("  Random split: NONE")
    print("  Test set: final evaluation only")
    print("  Classification threshold:", THRESHOLD)

    # --------------------------------------------------------
    # 1. Input checks
    # --------------------------------------------------------
    required_files = [
        TRAIN_PCA, VAL_PCA, TEST_PCA,
        TRAIN_LLM, VAL_LLM, TEST_LLM,
    ]

    print("\n[1] Checking input files...")
    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        print(f"  [OK] {path}")

    # --------------------------------------------------------
    # 2. Load data
    # --------------------------------------------------------
    print("\n[2] Loading PCA and LLM datasets...")

    train_pca = pd.read_csv(TRAIN_PCA)
    val_pca = pd.read_csv(VAL_PCA)
    test_pca = pd.read_csv(TEST_PCA)

    train_llm = pd.read_csv(TRAIN_LLM)
    val_llm = pd.read_csv(VAL_LLM)
    test_llm = pd.read_csv(TEST_LLM)

    print(
        f"  PCA rows: train={len(train_pca)}, "
        f"validation={len(val_pca)}, test={len(test_pca)}"
    )
    print(
        f"  LLM rows: train={len(train_llm)}, "
        f"validation={len(val_llm)}, test={len(test_llm)}"
    )

    expected_sizes = {
        "train": 41998,
        "validation": 8998,
        "test": 9000,
    }

    actual_sizes = {
        "train": len(train_pca),
        "validation": len(val_pca),
        "test": len(test_pca),
    }

    if actual_sizes != expected_sizes:
        raise AssertionError(
            f"Unexpected split sizes: {actual_sizes}; expected {expected_sizes}"
        )

    if (
        len(train_llm) != expected_sizes["train"]
        or len(val_llm) != expected_sizes["validation"]
        or len(test_llm) != expected_sizes["test"]
    ):
        raise AssertionError("LLM split sizes do not match canonical split sizes.")

    # --------------------------------------------------------
    # 3. Integrity checks
    # --------------------------------------------------------
    print("\n[3] Validating IDs and labels...")
    validate_split_ids(
        train_pca, val_pca, test_pca,
        train_llm, val_llm, test_llm
    )

    y_train = validate_labels(train_pca, train_llm, "train")
    y_val = validate_labels(val_pca, val_llm, "validation")
    y_test = validate_labels(test_pca, test_llm, "test")

    print("  [OK] Train labels match.")
    print("  [OK] Validation labels match.")
    print("  [OK] Test labels match.")

    print(
        f"\n  Target prevalence:"
        f"\n    Train      : {y_train.mean():.4%}"
        f"\n    Validation : {y_val.mean():.4%}"
        f"\n    Test       : {y_test.mean():.4%}"
    )

    # --------------------------------------------------------
    # 4. Build LLM features
    # --------------------------------------------------------
    print("\n[4] Fitting LLM preprocessing on TRAIN only...")

    (
        X_train_llm,
        X_val_llm,
        X_test_llm,
        encoder,
        imputer,
    ) = fit_transform_llm(train_llm, val_llm, test_llm)

    # --------------------------------------------------------
    # 5. Build full three-way fusion
    # --------------------------------------------------------
    print("\n[5] Building full three-way feature fusion...")

    X_train = build_feature_matrix(train_pca, X_train_llm)
    X_val = build_feature_matrix(val_pca, X_val_llm)
    X_test = build_feature_matrix(test_pca, X_test_llm)

    print(f"  X_train: {X_train.shape}")
    print(f"  X_val  : {X_val.shape}")
    print(f"  X_test : {X_test.shape}")

    if X_train.shape != (41998, TOTAL_FEATURES):
        raise AssertionError(f"Unexpected X_train shape: {X_train.shape}")
    if X_val.shape != (8998, TOTAL_FEATURES):
        raise AssertionError(f"Unexpected X_val shape: {X_val.shape}")
    if X_test.shape != (9000, TOTAL_FEATURES):
        raise AssertionError(f"Unexpected X_test shape: {X_test.shape}")

    for name, matrix in [
        ("X_train", X_train),
        ("X_val", X_val),
        ("X_test", X_test),
    ]:
        if not np.isfinite(matrix).all():
            raise AssertionError(f"{name} contains NaN/Inf.")

    print(f"  [OK] {JIT_DIM} JIT + {CODEBERT_DIM} CodeBERT + "
          f"{LLM_DIM} LLM = {TOTAL_FEATURES} features.")
    print("  [OK] No NaN/Inf in fused matrices.")

    # --------------------------------------------------------
    # 6. Class imbalance
    # --------------------------------------------------------
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())

    scale_pos_weight = negatives / positives

    print("\n[6] Class imbalance:")
    print(f"  Train negatives: {negatives}")
    print(f"  Train positives: {positives}")
    print(f"  scale_pos_weight: {scale_pos_weight:.4f}")

    # --------------------------------------------------------
    # 7. Configure models
    # --------------------------------------------------------
    print("\n[7] Configuring base models...")

    rf = RandomForestClassifier(**RF_PARAMS)

    xgb = XGBClassifier(
        **XGB_BASE_PARAMS,
        scale_pos_weight=scale_pos_weight,
    )

    lgbm = LGBMClassifier(
        **LGBM_BASE_PARAMS,
        scale_pos_weight=scale_pos_weight,
    )

    print("  [OK] Random Forest")
    print("  [OK] XGBoost")
    print("  [OK] LightGBM")

    # --------------------------------------------------------
    # 8. Train base models on TRAIN only
    # --------------------------------------------------------
    print("\n[8] Training base models on TRAIN only...")

    print("  Training Random Forest...")
    rf.fit(X_train, y_train)

    print("  Training XGBoost...")
    xgb.fit(X_train, y_train)

    print("  Training LightGBM...")
    lgbm.fit(X_train, y_train)

    print("  [OK] All base models trained.")

    # --------------------------------------------------------
    # 9. Generate validation/test probabilities
    # --------------------------------------------------------
    print("\n[9] Generating base-model probabilities...")

    p_rf_val = rf.predict_proba(X_val)[:, 1]
    p_rf_test = rf.predict_proba(X_test)[:, 1]

    p_xgb_val = xgb.predict_proba(X_val)[:, 1]
    p_xgb_test = xgb.predict_proba(X_test)[:, 1]

    p_lgbm_val = lgbm.predict_proba(X_val)[:, 1]
    p_lgbm_test = lgbm.predict_proba(X_test)[:, 1]

    # --------------------------------------------------------
    # 10. Base model metrics
    # --------------------------------------------------------
    base_val_metrics = {
        "random_forest": metrics_from_probabilities(y_val, p_rf_val, THRESHOLD),
        "xgboost": metrics_from_probabilities(y_val, p_xgb_val, THRESHOLD),
        "lightgbm": metrics_from_probabilities(y_val, p_lgbm_val, THRESHOLD),
    }

    base_test_metrics = {
        "random_forest": metrics_from_probabilities(y_test, p_rf_test, THRESHOLD),
        "xgboost": metrics_from_probabilities(y_test, p_xgb_test, THRESHOLD),
        "lightgbm": metrics_from_probabilities(y_test, p_lgbm_test, THRESHOLD),
    }

    print("\n[10] Base-model validation metrics:")
    for name, m in base_val_metrics.items():
        print_metrics(name, m)

    print("\n[10] Base-model test metrics:")
    for name, m in base_test_metrics.items():
        print_metrics(name, m)

    # --------------------------------------------------------
    # 11. Select weights on VALIDATION ONLY
    # --------------------------------------------------------
    print("\n[11] Selecting blend weights on VALIDATION ONLY...")

    best_weights, all_candidates = select_blend_weights(
        y_val,
        p_rf_val,
        p_xgb_val,
        p_lgbm_val,
    )

    print("\n  Selected weights:")
    print(f"    Random Forest: {best_weights['rf_weight']:.2f}")
    print(f"    XGBoost      : {best_weights['xgb_weight']:.2f}")
    print(f"    LightGBM     : {best_weights['lightgbm_weight']:.2f}")
    print(
        f"    Sum          : "
        f"{best_weights['rf_weight'] + best_weights['xgb_weight'] + best_weights['lightgbm_weight']:.2f}"
    )

    print("\n  Validation metrics at selected weights:")
    print(f"    Accuracy : {best_weights['accuracy']:.6f}")
    print(f"    Precision: {best_weights['precision']:.6f}")
    print(f"    Recall   : {best_weights['recall']:.6f}")
    print(f"    F1       : {best_weights['f1']:.6f}")
    print(f"    MCC      : {best_weights['mcc']:.6f}")
    print(f"    ROC-AUC  : {best_weights['roc_auc']:.6f}")
    print(f"    PR-AUC   : {best_weights['pr_auc']:.6f}")

    # --------------------------------------------------------
    # 12. Apply selected weights
    # --------------------------------------------------------
    w_rf = best_weights["rf_weight"]
    w_xgb = best_weights["xgb_weight"]
    w_lgbm = best_weights["lightgbm_weight"]

    p_blend_val = (
        w_rf * p_rf_val
        + w_xgb * p_xgb_val
        + w_lgbm * p_lgbm_val
    )

    p_blend_test = (
        w_rf * p_rf_test
        + w_xgb * p_xgb_test
        + w_lgbm * p_lgbm_test
    )

    blend_val_metrics = metrics_from_probabilities(
        y_val, p_blend_val, THRESHOLD
    )
    blend_test_metrics = metrics_from_probabilities(
        y_test, p_blend_test, THRESHOLD
    )

    # --------------------------------------------------------
    # 13. Final blend evaluation
    # --------------------------------------------------------
    print("\n[12] FINAL WEIGHTED-BLEND RESULTS")

    print_metrics("Validation - Weighted Probability Blend", blend_val_metrics)
    print_metrics("Test - Weighted Probability Blend", blend_test_metrics)

    # --------------------------------------------------------
    # 14. Save predictions
    # --------------------------------------------------------
    print("\n[13] Saving predictions...")

    val_predictions = pd.DataFrame({
        ID_COL: val_pca[ID_COL].astype(str).values,
        TARGET: y_val,
        "rf_probability": p_rf_val,
        "xgb_probability": p_xgb_val,
        "lightgbm_probability": p_lgbm_val,
        "blend_probability": p_blend_val,
        "blend_prediction": (p_blend_val >= THRESHOLD).astype(int),
    })

    test_predictions = pd.DataFrame({
        ID_COL: test_pca[ID_COL].astype(str).values,
        TARGET: y_test,
        "rf_probability": p_rf_test,
        "xgb_probability": p_xgb_test,
        "lightgbm_probability": p_lgbm_test,
        "blend_probability": p_blend_test,
        "blend_prediction": (p_blend_test >= THRESHOLD).astype(int),
    })

    val_predictions.to_csv(
        OUTPUT_DIR / "validation_predictions.csv",
        index=False,
    )

    test_predictions.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
    )

    # Save the full validation grid so the selection is reproducible.
    pd.DataFrame(all_candidates).to_csv(
        OUTPUT_DIR / "blend_weight_search_validation.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 15. Save model configuration/results
    # --------------------------------------------------------
    config = {
        "experiment": "7F",
        "name": "Weighted Probability Blending",
        "feature_configuration": {
            "jit_features": JIT_DIM,
            "codebert_pca_features": CODEBERT_DIM,
            "codebert_source_dim": CODEBERT_SOURCE_DIM,
            "llm_features": LLM_DIM,
            "total_features": TOTAL_FEATURES,
        },
        "split": {
            "policy": "project-wise chronological 70/15/15",
            "train_rows": len(train_pca),
            "validation_rows": len(val_pca),
            "test_rows": len(test_pca),
        },
        "preprocessing": {
            "pca": "precomputed 384-D PCA; fitted on train only",
            "llm_one_hot": "fit on train only",
            "llm_imputation": "median; fit on train only",
            "resampling": False,
            "random_split": False,
        },
        "models": {
            "random_forest": RF_PARAMS,
            "xgboost": {
                **XGB_BASE_PARAMS,
                "scale_pos_weight": scale_pos_weight,
            },
            "lightgbm": {
                **LGBM_BASE_PARAMS,
                "scale_pos_weight": scale_pos_weight,
            },
        },
        "blending": {
            "method": "weighted probability average",
            "grid_step": BLEND_STEP,
            "selection_split": "validation",
            "selection_criteria": [
                "F1",
                "MCC",
                "ROC-AUC",
                "PR-AUC",
            ],
            "threshold": THRESHOLD,
            "selected_weights": {
                "random_forest": w_rf,
                "xgboost": w_xgb,
                "lightgbm": w_lgbm,
            },
        },
        "metrics": {
            "validation": blend_val_metrics,
            "test": blend_test_metrics,
            "base_validation": base_val_metrics,
            "base_test": base_test_metrics,
        },
    }

    with open(OUTPUT_DIR / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "validation": blend_val_metrics,
                "test": blend_test_metrics,
                "base_validation": base_val_metrics,
                "base_test": base_test_metrics,
                "selected_weights": {
                    "random_forest": w_rf,
                    "xgboost": w_xgb,
                    "lightgbm": w_lgbm,
                },
            },
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # 16. Save models
    # --------------------------------------------------------
    print("\n[14] Saving trained base models...")

    import joblib

    joblib.dump(
        rf,
        MODELS_DIR / "random_forest.joblib",
        compress=3,
    )
    joblib.dump(
        xgb,
        MODELS_DIR / "xgboost.joblib",
        compress=3,
    )
    joblib.dump(
        lgbm,
        MODELS_DIR / "lightgbm.joblib",
        compress=3,
    )

    # --------------------------------------------------------
    # 17. Final plots
    # --------------------------------------------------------
    print("\n[15] Generating final weighted-blend plots...")

    plot_roc_pr(y_val, p_blend_val, "Validation")
    plot_roc_pr(y_test, p_blend_test, "Test")

    expected_plots = [
        PLOTS_DIR / "weighted_blend_validation_roc.png",
        PLOTS_DIR / "weighted_blend_validation_pr.png",
        PLOTS_DIR / "weighted_blend_test_roc.png",
        PLOTS_DIR / "weighted_blend_test_pr.png",
    ]

    for plot_path in expected_plots:
        if not plot_path.exists():
            raise AssertionError(f"Expected plot was not generated: {plot_path}")
        print(f"  [OK] {plot_path.name}")

    # --------------------------------------------------------
    # 18. Final summary
    # --------------------------------------------------------
    print("\n" + "=" * 78)
    print("EXPERIMENT 7F COMPLETE")
    print("=" * 78)

    print("\nSelected validation weights:")
    print(f"  Random Forest = {w_rf:.2f}")
    print(f"  XGBoost       = {w_xgb:.2f}")
    print(f"  LightGBM      = {w_lgbm:.2f}")

    print("\nFinal test metrics:")
    print(
        f"  Accuracy : {blend_test_metrics['accuracy']:.6f}\n"
        f"  Precision: {blend_test_metrics['precision']:.6f}\n"
        f"  Recall   : {blend_test_metrics['recall']:.6f}\n"
        f"  F1       : {blend_test_metrics['f1']:.6f}\n"
        f"  MCC      : {blend_test_metrics['mcc']:.6f}\n"
        f"  ROC-AUC  : {blend_test_metrics['roc_auc']:.6f}\n"
        f"  PR-AUC   : {blend_test_metrics['pr_auc']:.6f}"
    )
    print(f"  CM       : {blend_test_metrics['confusion_matrix']}")

    print("\nOutput directory:")
    print(f"  {OUTPUT_DIR}")

    print("\n[OK] Test set was not used for weight selection.")
    print("[OK] All integrity checks passed.")
    print("[OK] Experiment 7F is ready for comparison with 7A-7E.")


if __name__ == "__main__":
    main()
