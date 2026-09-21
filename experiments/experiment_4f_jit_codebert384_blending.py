"""
EXPERIMENT 4F: JIT + FULL CODEBERT + THREE-MODEL WEIGHTED BLENDING
=================================================================

Purpose
-------
Directly compare the JIT + CodeBERT representation against Experiment 7F.

Feature representation:
    12 JIT + 384 CodeBERT PCA = 396 features

Models:
    1. Random Forest
    2. XGBoost
    3. LightGBM

Ensemble:
    Weighted probability blending
    - weights selected on VALIDATION ONLY
    - grid step = 0.05
    - RF + XGB + LightGBM weights sum to 1.0
    - selection: F1 -> MCC -> ROC-AUC -> PR-AUC

Comparison with Experiment 7F
-----------------------------
4F:
    12 JIT + 384 CodeBERT = 396 features

7F:
    12 JIT + 384 CodeBERT + 51 LLM = 447 features

Therefore, the only feature-level difference between 4F and 7F
is the presence of the 51 LLM reasoning features.

Methodology
-----------
- Project-wise chronological 70/15/15 split
- PCA already fitted on TRAIN only
- Use the existing 384-D PCA representation
- No PCA recomputation
- No resampling
- No random split
- Models trained on TRAIN only
- Blend weights selected using VALIDATION only
- TEST used only for final evaluation
- Threshold = 0.5
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PCA_DIR = ROOT / "results" / "pca"

TRAIN_PATH = PCA_DIR / "train_pca384.csv"
VAL_PATH = PCA_DIR / "validation_pca384.csv"
TEST_PATH = PCA_DIR / "test_pca384.csv"

OUTPUT_DIR = ROOT / "results" / "experiment_4f_jit_codebert384_blending"
MODELS_DIR = OUTPUT_DIR / "models"
PLOTS_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXPERIMENT CONSTANTS
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
ID_COL = "commit_id"

CODEBERT_DIM = 384
JIT_DIM = 12
TOTAL_FEATURES = 396

THRESHOLD = 0.50
BLEND_STEP = 0.05

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


# ============================================================
# HELPERS
# ============================================================

def normalize_target(series):
    """Normalize buggy labels to integer {0,1}."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")

        if values.isna().any():
            raise ValueError("Target contains NaN after numeric conversion.")

        unique = set(values.unique().tolist())

        if not unique.issubset({0, 1}):
            raise ValueError(
                f"Unexpected numeric target values: {sorted(unique)}"
            )

        return values.astype(int)

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

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    if normalized.isna().any():
        bad = series[normalized.isna()].astype(str).unique()[:10]
        raise ValueError(f"Unknown target labels: {bad}")

    return normalized.astype(int)


def parse_embedding(value, expected_dim=384):
    """Parse an embedding stored as JSON/list/string."""
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=np.float32)
    else:
        text = str(value).strip()

        try:
            arr = np.asarray(
                json.loads(text),
                dtype=np.float32,
            )
        except Exception:
            text = text.strip("[]")
            arr = np.fromstring(
                text,
                sep=",",
                dtype=np.float32,
            )

    arr = arr.reshape(-1)

    if len(arr) != expected_dim:
        raise ValueError(
            f"Expected {expected_dim}-D embedding, got {len(arr)}"
        )

    return arr


def extract_codebert(df):
    """Extract the existing 384-D train-only-PCA representation."""
    if "embedding" not in df.columns:
        raise KeyError(
            "Expected 'embedding' column in PCA split file."
        )

    matrix = np.vstack(
        [
            parse_embedding(value, CODEBERT_DIM)
            for value in df["embedding"].values
        ]
    ).astype(np.float32)

    if matrix.shape[1] != CODEBERT_DIM:
        raise AssertionError(
            f"Expected {CODEBERT_DIM} CodeBERT features, "
            f"got {matrix.shape[1]}"
        )

    return matrix


def build_features(df):
    """Build 12 JIT + 384 CodeBERT = 396 features."""
    missing = [
        feature
        for feature in JIT_FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise KeyError(f"Missing JIT features: {missing}")

    jit = (
        df[JIT_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )

    codebert = extract_codebert(df)

    X = np.hstack([jit, codebert]).astype(np.float32)

    if X.shape[1] != TOTAL_FEATURES:
        raise AssertionError(
            f"Expected {TOTAL_FEATURES} features, "
            f"got {X.shape[1]}"
        )

    return X


def validate_ids_and_splits(train, val, test):
    """Validate IDs and ensure chronological split separation."""
    for name, df in [
        ("train", train),
        ("validation", val),
        ("test", test),
    ]:
        if ID_COL not in df.columns:
            raise KeyError(f"{ID_COL} missing from {name}.")

        if df[ID_COL].duplicated().any():
            raise AssertionError(
                f"Duplicate commit IDs found in {name}."
            )

    train_ids = set(train[ID_COL].astype(str))
    val_ids = set(val[ID_COL].astype(str))
    test_ids = set(test[ID_COL].astype(str))

    if train_ids & val_ids:
        raise AssertionError("Train/validation ID overlap detected.")

    if train_ids & test_ids:
        raise AssertionError("Train/test ID overlap detected.")

    if val_ids & test_ids:
        raise AssertionError("Validation/test ID overlap detected.")

    print("  [OK] Commit IDs unique.")
    print("  [OK] Cross-split commit ID overlap = 0.")


def validate_labels(train, val, test):
    """Validate and return labels."""
    y_train = normalize_target(train[TARGET]).to_numpy()
    y_val = normalize_target(val[TARGET]).to_numpy()
    y_test = normalize_target(test[TARGET]).to_numpy()

    return y_train, y_val, y_test


def metrics_from_probabilities(
    y_true,
    probabilities,
    threshold=0.5,
):
    """Calculate threshold and ranking metrics."""
    predictions = (
        probabilities >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        predictions,
    )

    return {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                y_true,
                predictions,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "confusion_matrix": cm.tolist(),
    }


def print_metrics(name, metrics):
    print(f"\n{name}")
    print(
        f"  Accuracy : {metrics['accuracy']:.6f}"
    )
    print(
        f"  Precision: {metrics['precision']:.6f}"
    )
    print(
        f"  Recall   : {metrics['recall']:.6f}"
    )
    print(
        f"  F1       : {metrics['f1']:.6f}"
    )
    print(
        f"  MCC      : {metrics['mcc']:.6f}"
    )
    print(
        f"  ROC-AUC  : {metrics['roc_auc']:.6f}"
    )
    print(
        f"  PR-AUC   : {metrics['pr_auc']:.6f}"
    )
    print(
        f"  CM       : {metrics['confusion_matrix']}"
    )


def generate_weight_grid(step=0.05):
    """
    Generate RF/XGB/LGBM weights that sum to 1.0.
    """
    n = int(round(1.0 / step))
    weights = []

    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j

            weights.append(
                (
                    i / n,
                    j / n,
                    k / n,
                )
            )

    return weights


def select_blend_weights(
    y_val,
    p_rf,
    p_xgb,
    p_lgbm,
):
    """
    Select weights using validation only.

    Primary:
        F1

    Tie-breakers:
        MCC
        ROC-AUC
        PR-AUC
    """
    candidates = []

    for (
        rf_weight,
        xgb_weight,
        lgbm_weight,
    ) in generate_weight_grid(BLEND_STEP):

        blend_probability = (
            rf_weight * p_rf
            + xgb_weight * p_xgb
            + lgbm_weight * p_lgbm
        )

        metrics = metrics_from_probabilities(
            y_val,
            blend_probability,
            THRESHOLD,
        )

        candidates.append(
            {
                "rf_weight": rf_weight,
                "xgb_weight": xgb_weight,
                "lightgbm_weight": lgbm_weight,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "mcc": metrics["mcc"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
            }
        )

    candidates.sort(
        key=lambda row: (
            row["f1"],
            row["mcc"],
            row["roc_auc"],
            row["pr_auc"],
        ),
        reverse=True,
    )

    return candidates[0], candidates


def plot_roc_pr(
    y_true,
    probabilities,
    split_name,
):
    """Generate final blend ROC and PR plots."""
    fpr, tpr, _ = roc_curve(
        y_true,
        probabilities,
    )

    precision, recall, _ = precision_recall_curve(
        y_true,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"Weighted Blend (AUC = {roc_auc:.4f})",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        f"Experiment 4F - Weighted Blend ROC ({split_name})"
    )
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / f"weighted_blend_{split_name.lower()}_roc.png",
        dpi=300,
    )

    plt.close()

    plt.figure(figsize=(7, 6))

    plt.plot(
        recall,
        precision,
        label=f"Weighted Blend (AP = {pr_auc:.4f})",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        f"Experiment 4F - Weighted Blend PR ({split_name})"
    )
    plt.legend(loc="lower left")
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / f"weighted_blend_{split_name.lower()}_pr.png",
        dpi=300,
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "EXPERIMENT 4F: JIT + FULL CODEBERT + "
        "THREE-MODEL WEIGHTED BLENDING"
    )
    print("=" * 78)

    print("\nMethodology:")
    print(
        "  Split: Project-wise chronological 70/15/15"
    )
    print(
        "  Features: 12 JIT + 384 CodeBERT = 396"
    )
    print(
        "  CodeBERT: existing 384-D PCA representation"
    )
    print(
        "  PCA fitting: TRAIN only"
    )
    print(
        "  Base models: Random Forest + XGBoost + LightGBM"
    )
    print(
        "  Ensemble: weighted probability blending"
    )
    print(
        "  Weight grid step:", BLEND_STEP
    )
    print(
        "  Weight selection: VALIDATION ONLY"
    )
    print(
        "  Selection: F1 -> MCC -> ROC-AUC -> PR-AUC"
    )
    print(
        "  Resampling: NONE"
    )
    print(
        "  Random split: NONE"
    )
    print(
        "  Test set: final evaluation only"
    )
    print(
        "  Classification threshold:", THRESHOLD
    )

    # --------------------------------------------------------
    # 1. Input files
    # --------------------------------------------------------

    print("\n[1] Checking input files...")

    for path in [
        TRAIN_PATH,
        VAL_PATH,
        TEST_PATH,
    ]:

        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

        print(f"  [OK] {path}")

    # --------------------------------------------------------
    # 2. Load datasets
    # --------------------------------------------------------

    print("\n[2] Loading chronological PCA splits...")

    train = pd.read_csv(TRAIN_PATH)
    val = pd.read_csv(VAL_PATH)
    test = pd.read_csv(TEST_PATH)

    print(
        f"  Rows: train={len(train)}, "
        f"validation={len(val)}, "
        f"test={len(test)}"
    )

    expected_sizes = {
        "train": 41998,
        "validation": 8998,
        "test": 9000,
    }

    actual_sizes = {
        "train": len(train),
        "validation": len(val),
        "test": len(test),
    }

    if actual_sizes != expected_sizes:
        raise AssertionError(
            f"Unexpected split sizes: {actual_sizes}; "
            f"expected {expected_sizes}"
        )

    # --------------------------------------------------------
    # 3. Integrity checks
    # --------------------------------------------------------

    print("\n[3] Validating IDs and labels...")

    validate_ids_and_splits(
        train,
        val,
        test,
    )

    y_train, y_val, y_test = validate_labels(
        train,
        val,
        test,
    )

    print("  [OK] Train labels valid.")
    print("  [OK] Validation labels valid.")
    print("  [OK] Test labels valid.")

    print(
        "\n  Target prevalence:"
        f"\n    Train      : {y_train.mean():.4%}"
        f"\n    Validation : {y_val.mean():.4%}"
        f"\n    Test       : {y_test.mean():.4%}"
    )

    # --------------------------------------------------------
    # 4. Build 396-dimensional representation
    # --------------------------------------------------------

    print(
        "\n[4] Building JIT + FULL 384-D CodeBERT representation..."
    )

    X_train = build_features(train)
    X_val = build_features(val)
    X_test = build_features(test)

    print(
        f"  X_train: {X_train.shape}"
    )
    print(
        f"  X_val  : {X_val.shape}"
    )
    print(
        f"  X_test : {X_test.shape}"
    )

    if X_train.shape != (41998, 396):
        raise AssertionError(
            f"Unexpected train shape: {X_train.shape}"
        )

    if X_val.shape != (8998, 396):
        raise AssertionError(
            f"Unexpected validation shape: {X_val.shape}"
        )

    if X_test.shape != (9000, 396):
        raise AssertionError(
            f"Unexpected test shape: {X_test.shape}"
        )

    for name, matrix in [
        ("X_train", X_train),
        ("X_val", X_val),
        ("X_test", X_test),
    ]:

        if not np.isfinite(matrix).all():
            raise AssertionError(
                f"{name} contains NaN/Inf."
            )

    print(
        "  [OK] 12 JIT + 384 CodeBERT = 396 features."
    )
    print(
        "  [OK] No NaN/Inf in feature matrices."
    )

    # --------------------------------------------------------
    # 5. Class imbalance
    # --------------------------------------------------------

    print("\n[5] Class imbalance...")

    negatives = int(
        (y_train == 0).sum()
    )

    positives = int(
        (y_train == 1).sum()
    )

    scale_pos_weight = (
        negatives / positives
    )

    print(
        f"  Train negatives: {negatives}"
    )

    print(
        f"  Train positives: {positives}"
    )

    print(
        f"  scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    # --------------------------------------------------------
    # 6. Configure models
    # --------------------------------------------------------

    print("\n[6] Configuring models...")

    rf = RandomForestClassifier(
        **RF_PARAMS
    )

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
    # 7. Train base models
    # --------------------------------------------------------

    print(
        "\n[7] Training base models on TRAIN only..."
    )

    print(
        "  Training Random Forest..."
    )

    rf.fit(
        X_train,
        y_train,
    )

    print(
        "  Training XGBoost..."
    )

    xgb.fit(
        X_train,
        y_train,
    )

    print(
        "  Training LightGBM..."
    )

    lgbm.fit(
        X_train,
        y_train,
    )

    print(
        "  [OK] All base models trained."
    )

    # --------------------------------------------------------
    # 8. Generate probabilities
    # --------------------------------------------------------

    print(
        "\n[8] Generating validation/test probabilities..."
    )

    p_rf_val = rf.predict_proba(
        X_val
    )[:, 1]

    p_rf_test = rf.predict_proba(
        X_test
    )[:, 1]

    p_xgb_val = xgb.predict_proba(
        X_val
    )[:, 1]

    p_xgb_test = xgb.predict_proba(
        X_test
    )[:, 1]

    p_lgbm_val = lgbm.predict_proba(
        X_val
    )[:, 1]

    p_lgbm_test = lgbm.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # 9. Base-model metrics
    # --------------------------------------------------------

    print(
        "\n[9] Base-model validation metrics:"
    )

    rf_val_metrics = metrics_from_probabilities(
        y_val,
        p_rf_val,
        THRESHOLD,
    )

    xgb_val_metrics = metrics_from_probabilities(
        y_val,
        p_xgb_val,
        THRESHOLD,
    )

    lgbm_val_metrics = metrics_from_probabilities(
        y_val,
        p_lgbm_val,
        THRESHOLD,
    )

    print_metrics(
        "Random Forest - Validation",
        rf_val_metrics,
    )

    print_metrics(
        "XGBoost - Validation",
        xgb_val_metrics,
    )

    print_metrics(
        "LightGBM - Validation",
        lgbm_val_metrics,
    )

    print(
        "\n[9] Base-model test metrics:"
    )

    rf_test_metrics = metrics_from_probabilities(
        y_test,
        p_rf_test,
        THRESHOLD,
    )

    xgb_test_metrics = metrics_from_probabilities(
        y_test,
        p_xgb_test,
        THRESHOLD,
    )

    lgbm_test_metrics = metrics_from_probabilities(
        y_test,
        p_lgbm_test,
        THRESHOLD,
    )

    print_metrics(
        "Random Forest - Test",
        rf_test_metrics,
    )

    print_metrics(
        "XGBoost - Test",
        xgb_test_metrics,
    )

    print_metrics(
        "LightGBM - Test",
        lgbm_test_metrics,
    )

    # --------------------------------------------------------
    # 10. Validation-only weight selection
    # --------------------------------------------------------

    print(
        "\n[10] Selecting blend weights "
        "on VALIDATION ONLY..."
    )

    best_weights, all_candidates = (
        select_blend_weights(
            y_val,
            p_rf_val,
            p_xgb_val,
            p_lgbm_val,
        )
    )

    w_rf = best_weights[
        "rf_weight"
    ]

    w_xgb = best_weights[
        "xgb_weight"
    ]

    w_lgbm = best_weights[
        "lightgbm_weight"
    ]

    print(
        "\n  Selected weights:"
    )

    print(
        f"    Random Forest: {w_rf:.2f}"
    )

    print(
        f"    XGBoost      : {w_xgb:.2f}"
    )

    print(
        f"    LightGBM     : {w_lgbm:.2f}"
    )

    print(
        f"    Sum          : "
        f"{w_rf + w_xgb + w_lgbm:.2f}"
    )

    # --------------------------------------------------------
    # 11. Apply blend
    # --------------------------------------------------------

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
        y_val,
        p_blend_val,
        THRESHOLD,
    )

    blend_test_metrics = metrics_from_probabilities(
        y_test,
        p_blend_test,
        THRESHOLD,
    )

    # --------------------------------------------------------
    # 12. Final results
    # --------------------------------------------------------

    print(
        "\n[11] FINAL WEIGHTED-BLEND RESULTS"
    )

    print_metrics(
        "Validation - Weighted Probability Blend",
        blend_val_metrics,
    )

    print_metrics(
        "Test - Weighted Probability Blend",
        blend_test_metrics,
    )

    # --------------------------------------------------------
    # 13. Save predictions
    # --------------------------------------------------------

    print(
        "\n[12] Saving predictions..."
    )

    validation_predictions = pd.DataFrame(
        {
            ID_COL: val[
                ID_COL
            ].astype(str).values,

            TARGET: y_val,

            "rf_probability":
                p_rf_val,

            "xgb_probability":
                p_xgb_val,

            "lightgbm_probability":
                p_lgbm_val,

            "blend_probability":
                p_blend_val,

            "blend_prediction":
                (
                    p_blend_val
                    >= THRESHOLD
                ).astype(int),
        }
    )

    test_predictions = pd.DataFrame(
        {
            ID_COL: test[
                ID_COL
            ].astype(str).values,

            TARGET: y_test,

            "rf_probability":
                p_rf_test,

            "xgb_probability":
                p_xgb_test,

            "lightgbm_probability":
                p_lgbm_test,

            "blend_probability":
                p_blend_test,

            "blend_prediction":
                (
                    p_blend_test
                    >= THRESHOLD
                ).astype(int),
        }
    )

    validation_predictions.to_csv(
        OUTPUT_DIR
        / "validation_predictions.csv",
        index=False,
    )

    test_predictions.to_csv(
        OUTPUT_DIR
        / "test_predictions.csv",
        index=False,
    )

    pd.DataFrame(
        all_candidates
    ).to_csv(
        OUTPUT_DIR
        / "blend_weight_search_validation.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 14. Save metrics/config
    # --------------------------------------------------------

    print(
        "\n[13] Saving metrics and configuration..."
    )

    base_validation = {
        "random_forest":
            rf_val_metrics,

        "xgboost":
            xgb_val_metrics,

        "lightgbm":
            lgbm_val_metrics,
    }

    base_test = {
        "random_forest":
            rf_test_metrics,

        "xgboost":
            xgb_test_metrics,

        "lightgbm":
            lgbm_test_metrics,
    }

    config = {
        "experiment": "4F",

        "name":
            "JIT + Full CodeBERT + "
            "Three-Model Weighted Probability Blending",

        "feature_configuration": {
            "jit_features": JIT_DIM,
            "codebert_pca_features": CODEBERT_DIM,
            "total_features": TOTAL_FEATURES,
            "llm_features": 0,
        },

        "comparison_with_7F": {
            "4F_features":
                "12 JIT + 384 CodeBERT = 396",

            "7F_features":
                "12 JIT + 384 CodeBERT + 51 LLM = 447",

            "feature_difference":
                "51 LLM reasoning features",
        },

        "split": {
            "policy":
                "project-wise chronological 70/15/15",

            "train_rows":
                len(train),

            "validation_rows":
                len(val),

            "test_rows":
                len(test),
        },

        "preprocessing": {
            "pca":
                "precomputed 384-D PCA; "
                "fitted on train only",

            "resampling":
                False,

            "random_split":
                False,
        },

        "models": {
            "random_forest":
                RF_PARAMS,

            "xgboost": {
                **XGB_BASE_PARAMS,
                "scale_pos_weight":
                    scale_pos_weight,
            },

            "lightgbm": {
                **LGBM_BASE_PARAMS,
                "scale_pos_weight":
                    scale_pos_weight,
            },
        },

        "blending": {
            "method":
                "weighted probability average",

            "grid_step":
                BLEND_STEP,

            "selection_split":
                "validation",

            "selection_criteria": [
                "F1",
                "MCC",
                "ROC-AUC",
                "PR-AUC",
            ],

            "threshold":
                THRESHOLD,

            "selected_weights": {
                "random_forest":
                    w_rf,

                "xgboost":
                    w_xgb,

                "lightgbm":
                    w_lgbm,
            },
        },

        "metrics": {
            "validation":
                blend_val_metrics,

            "test":
                blend_test_metrics,

            "base_validation":
                base_validation,

            "base_test":
                base_test,
        },
    }

    with open(
        OUTPUT_DIR
        / "experiment_config.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=2,
        )

    with open(
        OUTPUT_DIR
        / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "validation":
                    blend_val_metrics,

                "test":
                    blend_test_metrics,

                "base_validation":
                    base_validation,

                "base_test":
                    base_test,

                "selected_weights": {
                    "random_forest":
                        w_rf,

                    "xgboost":
                        w_xgb,

                    "lightgbm":
                        w_lgbm,
                },
            },
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # 15. Save models
    # --------------------------------------------------------

    print(
        "\n[14] Saving trained base models..."
    )

    import joblib

    joblib.dump(
        rf,
        MODELS_DIR
        / "random_forest.joblib",
        compress=3,
    )

    joblib.dump(
        xgb,
        MODELS_DIR
        / "xgboost.joblib",
        compress=3,
    )

    joblib.dump(
        lgbm,
        MODELS_DIR
        / "lightgbm.joblib",
        compress=3,
    )

    # --------------------------------------------------------
    # 16. Plots
    # --------------------------------------------------------

    print(
        "\n[15] Generating final blend plots..."
    )

    plot_roc_pr(
        y_val,
        p_blend_val,
        "Validation",
    )

    plot_roc_pr(
        y_test,
        p_blend_test,
        "Test",
    )

    expected_plots = [
        PLOTS_DIR
        / "weighted_blend_validation_roc.png",

        PLOTS_DIR
        / "weighted_blend_validation_pr.png",

        PLOTS_DIR
        / "weighted_blend_test_roc.png",

        PLOTS_DIR
        / "weighted_blend_test_pr.png",
    ]

    for plot in expected_plots:

        if not plot.exists():
            raise AssertionError(
                f"Missing expected plot: {plot}"
            )

        print(
            f"  [OK] {plot.name}"
        )

    # --------------------------------------------------------
    # 17. Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print(
        "EXPERIMENT 4F COMPLETE"
    )
    print("=" * 78)

    print(
        "\nSelected validation weights:"
    )

    print(
        f"  Random Forest = {w_rf:.2f}"
    )

    print(
        f"  XGBoost       = {w_xgb:.2f}"
    )

    print(
        f"  LightGBM      = {w_lgbm:.2f}"
    )

    print(
        "\nFinal test metrics:"
    )

    print(
        f"  Accuracy : "
        f"{blend_test_metrics['accuracy']:.6f}"
    )

    print(
        f"  Precision: "
        f"{blend_test_metrics['precision']:.6f}"
    )

    print(
        f"  Recall   : "
        f"{blend_test_metrics['recall']:.6f}"
    )

    print(
        f"  F1       : "
        f"{blend_test_metrics['f1']:.6f}"
    )

    print(
        f"  MCC      : "
        f"{blend_test_metrics['mcc']:.6f}"
    )

    print(
        f"  ROC-AUC  : "
        f"{blend_test_metrics['roc_auc']:.6f}"
    )

    print(
        f"  PR-AUC   : "
        f"{blend_test_metrics['pr_auc']:.6f}"
    )

    print(
        f"  CM       : "
        f"{blend_test_metrics['confusion_matrix']}"
    )

    print(
        "\nOutput directory:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print(
        "\n[OK] Test set was not used "
        "for weight selection."
    )

    print(
        "[OK] All integrity checks passed."
    )

    print(
        "[OK] 4F is directly comparable "
        "with 7F."
    )


if __name__ == "__main__":
    main()
