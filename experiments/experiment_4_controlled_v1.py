"""Controlled Experiment 4: JIT + CodeBERT Fusion Benchmark.

Methodology:
- Strict chronological 3-way split:
    Train (6,000 rows, 60%) -> Validation (2,000 rows, 20%) -> Untouched Final Test (2,000 rows, 20%)
- Compares:
    1. JIT-only baseline (Experiment 1 reference)
    2. CodeBERT-only baseline (Experiment 2 reference)
    3. 4a: Early Fusion (JIT 13 + CodeBERT 384 = 397 features)
    4. 4b: Top-K Feature Selection (JIT 13 + Top-25 / Top-50 CodeBERT PCA components)
    5. 4c: Pure Top-25 Mutual Information features selected on Train
    6. 4d: Stacking Ensemble (Logistic Regression meta-learner combining JIT + CodeBERT predictions)
    7. 4e: Probability Blend (optimal alpha blend tuned on Validation)
- Threshold tuning & calibration locked strictly on Validation.
- Evaluates final test period exactly once.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "results" / "data" / "java_only_frozen_sample_dataset.csv"
EXP1_SUMMARY_PATH = PROJECT_ROOT / "results" / "java_only_experiment_1_controlled_v1" / "metrics" / "summary.json"
EXP2_SUMMARY_PATH = PROJECT_ROOT / "results" / "java_only_experiment_2_controlled_v1" / "metrics" / "summary.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "java_only_experiment_4_controlled_v1"

JIT_FEATURES: List[str] = [
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

PCA_COLS = [f"pca_{i}" for i in range(1, 385)]


def ensure_dirs() -> None:
    for sub in ["metrics", "predictions", "tables", "plots"]:
        (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)


def parse_author_date(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    numeric = pd.to_numeric(series, errors="coerce")
    if int(numeric.notna().sum()) >= max(1, int(len(series) * 0.1)):
        max_val = int(numeric.abs().max(skipna=True))
        if max_val > 10**15:
            unit = "ns"
        elif max_val > 10**12:
            unit = "ms"
        else:
            unit = "s"
        return pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)
    return pd.to_datetime(series, errors="coerce", utc=True)


def compute_metrics_dict(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(round(threshold, 4)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "predicted_positive_rate": float(np.mean(y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    candidates: np.ndarray,
    metric_key: str = "f1",
) -> Tuple[float, Dict[str, Any]]:
    best_threshold = 0.50
    best_val = -1.0
    best_metrics = None
    for t in candidates:
        m = compute_metrics_dict(y_true, y_prob, float(t))
        if m[metric_key] > best_val:
            best_val = m[metric_key]
            best_threshold = float(t)
            best_metrics = m
    return best_threshold, best_metrics or compute_metrics_dict(y_true, y_prob, 0.50)


def train_xgb(x_train: pd.DataFrame, y_train: pd.Series, spw: float, random_state: int = 42) -> XGBClassifier:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        learning_rate=0.1,
        n_estimators=100,
        scale_pos_weight=spw,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return model


def plot_diagnostics(
    y_test: np.ndarray,
    predictions_dict: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for name, data in predictions_dict.items():
        score = data["score"]
        fpr, tpr, _ = roc_curve(y_test, score)
        prec, rec, _ = precision_recall_curve(y_test, score)
        roc_auc = roc_auc_score(y_test, score)
        pr_auc = average_precision_score(y_test, score)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
        axes[1].plot(rec, prec, label=f"{name} (PR-AUC={pr_auc:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", label="Chance")
    axes[0].set_title("Test ROC Curves: JIT vs CodeBERT vs Fusion")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(
        y=np.mean(y_test),
        color="k",
        linestyle="--",
        label=f"Prevalence ({np.mean(y_test)*100:.1f}%)",
    )
    axes[1].set_title("Test PR Curves: JIT vs CodeBERT vs Fusion")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "plots" / "roc_pr_curves.png", dpi=200)
    plt.close(fig)


def run_experiment() -> Dict[str, Any]:
    ensure_dirs()

    # Load frozen dataset
    logger.info("Loading dataset from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df["author_date"] = parse_author_date(df["author_date"])
    df = df.sort_values("author_date", kind="mergesort").reset_index(drop=True)

    if "embedding" in df.columns and not all(c in df.columns for c in PCA_COLS):
        logger.info("Expanding 'embedding' JSON column into %d PCA columns", len(PCA_COLS))
        embs = np.vstack([json.loads(s) for s in df["embedding"]])
        emb_df = pd.DataFrame(embs, columns=PCA_COLS)
        df = pd.concat([df.drop(columns=["embedding"]), emb_df], axis=1)

    all_features = JIT_FEATURES + PCA_COLS
    keep_cols = ["commit_id", "project", "author_date", "buggy"] + all_features
    df = df[keep_cols].dropna(subset=["author_date"]).copy()
    df["buggy"] = df["buggy"].astype(int)

    # 3-way chronological split
    n = len(df)
    test_start = int(n * 0.8)
    dev_df = df.iloc[:test_start].copy()
    test_df = df.iloc[test_start:].copy()

    val_start = int(len(dev_df) * 0.75)
    train_df = dev_df.iloc[:val_start].copy()
    val_df = dev_df.iloc[val_start:].copy()

    train_neg = int((train_df["buggy"] == 0).sum())
    train_pos = int((train_df["buggy"] == 1).sum())
    val_pos = int((val_df["buggy"] == 1).sum())
    test_pos = int((test_df["buggy"] == 1).sum())

    spw = float(train_neg / train_pos)

    logger.info("=== Splits and Class Prevalences ===")
    logger.info("Train: %d (Pos: %d, %.2f%%) | SPW: %.4f", len(train_df), train_pos, train_pos/len(train_df)*100, spw)
    logger.info("Val:   %d (Pos: %d, %.2f%%)", len(val_df), val_pos, val_pos/len(val_df)*100)
    logger.info("Test:  %d (Pos: %d, %.2f%%)", len(test_df), test_pos, test_pos/len(test_df)*100)

    y_train = train_df["buggy"]
    y_val = val_df["buggy"].to_numpy()
    y_test = test_df["buggy"].to_numpy()

    # Step 1: Base Single-Modality Models
    # A) JIT-Only Model
    logger.info("Training JIT-Only XGBoost...")
    model_jit = train_xgb(train_df[JIT_FEATURES], y_train, spw=spw)
    val_score_jit = model_jit.predict_proba(val_df[JIT_FEATURES])[:, 1]
    test_score_jit = model_jit.predict_proba(test_df[JIT_FEATURES])[:, 1]

    # B) CodeBERT-Only Model
    logger.info("Training CodeBERT-Only XGBoost...")
    model_codebert = train_xgb(train_df[PCA_COLS], y_train, spw=spw)
    val_score_cb = model_codebert.predict_proba(val_df[PCA_COLS])[:, 1]
    test_score_cb = model_codebert.predict_proba(test_df[PCA_COLS])[:, 1]

    # Step 2: Fusion Models
    # Variant 4a: Early Fusion (All JIT + All 384 PCA = 397 features)
    logger.info("Training 4a Early Fusion (All 397 features)...")
    early_features = JIT_FEATURES + PCA_COLS
    model_4a = train_xgb(train_df[early_features], y_train, spw=spw)
    val_score_4a = model_4a.predict_proba(val_df[early_features])[:, 1]
    test_score_4a = model_4a.predict_proba(test_df[early_features])[:, 1]

    # Variant 4b-1: JIT + Top-25 CodeBERT PCA (pca_1 .. pca_25)
    logger.info("Training 4b-1 (JIT 13 + First 25 PCA = 38 features)...")
    top25_pca = PCA_COLS[:25]
    features_4b1 = JIT_FEATURES + top25_pca
    model_4b1 = train_xgb(train_df[features_4b1], y_train, spw=spw)
    val_score_4b1 = model_4b1.predict_proba(val_df[features_4b1])[:, 1]
    test_score_4b1 = model_4b1.predict_proba(test_df[features_4b1])[:, 1]

    # Variant 4b-2: JIT + Top-50 CodeBERT PCA (pca_1 .. pca_50)
    logger.info("Training 4b-2 (JIT 13 + First 50 PCA = 63 features)...")
    top50_pca = PCA_COLS[:50]
    features_4b2 = JIT_FEATURES + top50_pca
    model_4b2 = train_xgb(train_df[features_4b2], y_train, spw=spw)
    val_score_4b2 = model_4b2.predict_proba(val_df[features_4b2])[:, 1]
    test_score_4b2 = model_4b2.predict_proba(test_df[features_4b2])[:, 1]

    # Variant 4c: Mutual Information Selected Top-25 Features on Train
    logger.info("Computing Mutual Information on Train...")
    mi_scores = mutual_info_classif(train_df[all_features], y_train, random_state=42)
    mi_ranked = [feat for _, feat in sorted(zip(mi_scores, all_features), reverse=True)]
    top25_mi = mi_ranked[:25]
    logger.info("Top 25 MI features: %s", top25_mi)
    model_4c = train_xgb(train_df[top25_mi], y_train, spw=spw)
    val_score_4c = model_4c.predict_proba(val_df[top25_mi])[:, 1]
    test_score_4c = model_4c.predict_proba(test_df[top25_mi])[:, 1]

    # Variant 4d: Stacking Ensemble (Late Fusion via Logistic Regression Meta-Learner)
    logger.info("Training 4d Stacking Ensemble...")
    val_meta_X = np.column_stack([val_score_jit, val_score_cb])
    test_meta_X = np.column_stack([test_score_jit, test_score_cb])
    meta_learner = LogisticRegression(solver="lbfgs", random_state=42)
    meta_learner.fit(val_meta_X, y_val)
    val_score_4d = meta_learner.predict_proba(val_meta_X)[:, 1]
    test_score_4d = meta_learner.predict_proba(test_meta_X)[:, 1]

    # Variant 4e: Optimal Probability Blend (alpha * JIT + (1-alpha) * CodeBERT)
    best_alpha = 0.5
    best_blend_f1 = -1.0
    for alpha in np.linspace(0.1, 0.9, 17):
        blend_val = alpha * val_score_jit + (1 - alpha) * val_score_cb
        _, bm = find_optimal_threshold(y_val, blend_val, np.linspace(0.01, 0.99, 99))
        if bm["f1"] > best_blend_f1:
            best_blend_f1 = bm["f1"]
            best_alpha = float(alpha)
    logger.info("Optimal validation probability blend alpha: %.2f (JIT weight)", best_alpha)
    val_score_4e = best_alpha * val_score_jit + (1 - best_alpha) * val_score_cb
    test_score_4e = best_alpha * test_score_jit + (1 - best_alpha) * test_score_cb

    # Assemble all variants
    variants = {
        "JIT_Only (Exp 1 Control)": {
            "val_score": val_score_jit,
            "test_score": test_score_jit,
            "desc": "JIT metrics only (13 features)",
        },
        "CodeBERT_Only (Exp 2 Control)": {
            "val_score": val_score_cb,
            "test_score": test_score_cb,
            "desc": "CodeBERT PCA features only (384 features)",
        },
        "4a_Early_Fusion_All": {
            "val_score": val_score_4a,
            "test_score": test_score_4a,
            "desc": "Early concatenation of 13 JIT + 384 CodeBERT PCA features (397 total)",
        },
        "4b1_Early_Fusion_Top25_PCA": {
            "val_score": val_score_4b1,
            "test_score": test_score_4b1,
            "desc": "13 JIT + First 25 CodeBERT PCA features (38 total)",
        },
        "4b2_Early_Fusion_Top50_PCA": {
            "val_score": val_score_4b2,
            "test_score": test_score_4b2,
            "desc": "13 JIT + First 50 CodeBERT PCA features (63 total)",
        },
        "4c_MI_Top25_Selected": {
            "val_score": val_score_4c,
            "test_score": test_score_4c,
            "desc": "Top-25 Mutual Information features selected on Train",
        },
        "4d_Stacking_Ensemble": {
            "val_score": val_score_4d,
            "test_score": test_score_4d,
            "desc": "Stacking ensemble (Logistic Regression meta-learner on JIT + CodeBERT scores)",
        },
        "4e_Probability_Blend": {
            "val_score": val_score_4e,
            "test_score": test_score_4e,
            "desc": f"Weighted score blend ({best_alpha:.2f}*JIT + {1-best_alpha:.2f}*CodeBERT)",
        },
    }

    threshold_grid = np.linspace(0.01, 0.99, 99)
    val_records = []
    test_records = []
    predictions_export = test_df[["commit_id", "project", "author_date"]].copy()
    predictions_export["buggy_true"] = y_test

    for name, data in variants.items():
        v_scores = data["val_score"]
        t_scores = data["test_score"]

        # Default 0.50 evaluation
        m_val_05 = compute_metrics_dict(y_val, v_scores, 0.50)
        m_test_05 = compute_metrics_dict(y_test, t_scores, 0.50)

        # Validation-Tuned threshold
        opt_thresh, m_val_opt = find_optimal_threshold(y_val, v_scores, threshold_grid, metric_key="f1")
        m_test_opt = compute_metrics_dict(y_test, t_scores, opt_thresh)

        data["opt_thresh"] = opt_thresh
        data["val_metrics_opt"] = m_val_opt
        data["test_metrics_opt"] = m_test_opt

        val_records.append({
            "variant": name,
            "description": data["desc"],
            "val_roc_auc": m_val_opt["roc_auc"],
            "val_pr_auc": m_val_opt["pr_auc"],
            "val_opt_threshold": opt_thresh,
            "val_precision": m_val_opt["precision"],
            "val_recall": m_val_opt["recall"],
            "val_f1": m_val_opt["f1"],
            "val_mcc": m_val_opt["mcc"],
            "val_fp": m_val_opt["fp"],
        })

        test_records.append({
            "variant": name,
            "description": data["desc"],
            "test_roc_auc": m_test_opt["roc_auc"],
            "test_pr_auc": m_test_opt["pr_auc"],
            "test_opt_threshold": opt_thresh,
            "test_precision": m_test_opt["precision"],
            "test_recall": m_test_opt["recall"],
            "test_f1": m_test_opt["f1"],
            "test_mcc": m_test_opt["mcc"],
            "test_tp": m_test_opt["tp"],
            "test_fp": m_test_opt["fp"],
            "test_fn": m_test_opt["fn"],
            "test_tn": m_test_opt["tn"],
            "test_f1_at_05": m_test_05["f1"],
            "test_prec_at_05": m_test_05["precision"],
            "test_rec_at_05": m_test_05["recall"],
        })

        predictions_export[f"score_{name}"] = t_scores
        predictions_export[f"pred_{name}"] = (t_scores >= opt_thresh).astype(int)

    val_df_out = pd.DataFrame(val_records)
    test_df_out = pd.DataFrame(test_records)

    val_df_out.to_csv(OUTPUT_DIR / "tables" / "validation_fusion_comparison.csv", index=False)
    test_df_out.to_csv(OUTPUT_DIR / "tables" / "final_test_fusion_comparison.csv", index=False)
    predictions_export.to_csv(OUTPUT_DIR / "predictions" / "final_test_predictions.csv", index=False)

    logger.info("=== Validation Fusion Comparison ===")
    print(val_df_out[["variant", "val_roc_auc", "val_pr_auc", "val_opt_threshold", "val_precision", "val_recall", "val_f1", "val_mcc"]].to_string(index=False))

    logger.info("=== Final Test Fusion Comparison ===")
    print(test_df_out[["variant", "test_roc_auc", "test_pr_auc", "test_precision", "test_recall", "test_f1", "test_mcc", "test_tp", "test_fp"]].to_string(index=False))

    # Diagnostics plot
    diag_dict = {
        "JIT-Only": {"score": variants["JIT_Only (Exp 1 Control)"]["test_score"]},
        "CodeBERT-Only": {"score": variants["CodeBERT_Only (Exp 2 Control)"]["test_score"]},
        "4a Early Fusion": {"score": variants["4a_Early_Fusion_All"]["test_score"]},
        "4b1 JIT+Top25": {"score": variants["4b1_Early_Fusion_Top25_PCA"]["test_score"]},
        "4d Stacking": {"score": variants["4d_Stacking_Ensemble"]["test_score"]},
        "4e Blend": {"score": variants["4e_Probability_Blend"]["test_score"]},
    }
    plot_diagnostics(y_test, diag_dict, OUTPUT_DIR)

    # Save summary JSON
    summary = {
        "experiment": "java_only_experiment_4_controlled_v1",
        "methodology": "Strict chronological split: Train (6,000) -> Val (2,000) -> Untouched Test (2,000)",
        "validation_results": val_records,
        "final_test_results": test_records,
    }
    summary_path = OUTPUT_DIR / "metrics" / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved summary JSON to %s", summary_path)

    return summary


if __name__ == "__main__":
    run_experiment()
