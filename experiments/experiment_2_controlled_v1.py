"""Controlled Experiment 2: CodeBERT-only on Frozen Java Dataset.

Methodology:
- Strict chronological 3-way split:
    Train (6,000 rows, 60%) -> Validation (2,000 rows, 20%) -> Untouched Final Test (2,000 rows, 20%)
- Model architecture & features: CodeBERT PCA components (pca_1 .. pca_384).
- Models evaluated:
    1. XGBoost Unweighted (SPW=1.0)
    2. XGBoost Dynamic SPW (SPW = train_neg / train_pos)
    3. XGBoost Frozen Baseline SPW (SPW = 4.9084)
    4. Random Forest Balanced (class_weight="balanced", original Exp 2 baseline)
    5. Random Forest Unweighted (class_weight=None)
- Threshold-independent ranking quality (ROC-AUC, PR-AUC) assessed across all splits.
- Threshold tuning & Platt scaling calibration locked strictly on Validation.
- Final test period evaluated once to diagnose precision, recall, calibration, and discrimination.
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
from sklearn.ensemble import RandomForestClassifier
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
BASELINE_EXP2_SUMMARY = PROJECT_ROOT / "results" / "java_only_experiment_2" / "metrics" / "summary.json"
EXP1_CONTROLLED_SUMMARY = PROJECT_ROOT / "results" / "java_only_experiment_1_controlled_v1" / "metrics" / "summary.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "java_only_experiment_2_controlled_v1"

PCA_COLS = [f"pca_{i}" for i in range(1, 385)]


def ensure_dirs() -> None:
    for sub in ["metrics", "predictions", "tables", "plots", "models"]:
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


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob > lo) & (y_prob <= hi)
        if i == 0:
            mask = (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / n) * abs(conf - acc)
    return float(ece)


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
        "ece": float(expected_calibration_error(y_true, y_prob, bins=10)),
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
) -> Tuple[float, Dict[str, Any], pd.DataFrame]:
    records = []
    for t in candidates:
        m = compute_metrics_dict(y_true, y_prob, float(t))
        records.append(m)
    table = pd.DataFrame(records)
    best_row = table.sort_values(
        [metric_key, "mcc", "precision"], ascending=[False, False, False]
    ).iloc[0]
    return float(best_row["threshold"]), best_row.to_dict(), table


def plot_diagnostics(
    y_test: np.ndarray,
    predictions_dict: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> None:
    # 1. ROC & PR Curves
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
    axes[0].set_title("Test ROC Curves (CodeBERT-Only)")
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
    axes[1].set_title("Test Precision-Recall Curves (CodeBERT-Only)")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "plots" / "roc_pr_curves.png", dpi=200)
    plt.close(fig)

    # 2. Reliability / Calibration Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    for name, data in predictions_dict.items():
        score = data["score"]
        prob_true, prob_pred = calibration_curve(y_test, score, n_bins=10)
        brier = brier_score_loss(y_test, score)
        ece = expected_calibration_error(y_test, score, bins=10)
        ax.plot(
            prob_pred,
            prob_true,
            "s-",
            label=f"{name} (Brier={brier:.3f}, ECE={ece:.3f})",
        )

    ax.set_title("Reliability Diagrams (Final Test Set - CodeBERT)")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "plots" / "calibration_curves.png", dpi=200)
    plt.close(fig)

    # 3. Score Distributions
    fig, axes = plt.subplots(1, len(predictions_dict), figsize=(6 * len(predictions_dict), 5))
    if len(predictions_dict) == 1:
        axes = [axes]
    for ax, (name, data) in zip(axes, predictions_dict.items()):
        score = data["score"]
        ax.hist(score[y_test == 0], bins=25, alpha=0.6, label="Clean (0)", density=True, color="tab:blue")
        ax.hist(score[y_test == 1], bins=25, alpha=0.6, label="Buggy (1)", density=True, color="tab:red")
        ax.axvline(data["threshold"], color="black", linestyle="--", label=f"Threshold ({data['threshold']:.2f})")
        ax.set_title(f"Score Distribution: {name}")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "plots" / "score_distributions.png", dpi=200)
    plt.close(fig)


def run_experiment() -> Dict[str, Any]:
    ensure_dirs()

    # 1. Load frozen dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

    logger.info("Loading dataset from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df["author_date"] = parse_author_date(df["author_date"])
    df = df.sort_values("author_date", kind="mergesort").reset_index(drop=True)

    # Expand embeddings into 384 PCA features
    if "embedding" in df.columns and not all(c in df.columns for c in PCA_COLS):
        logger.info("Expanding 'embedding' JSON column into %d PCA columns", len(PCA_COLS))
        embs = np.vstack([json.loads(s) for s in df["embedding"]])
        emb_df = pd.DataFrame(embs, columns=PCA_COLS)
        df = pd.concat([df.drop(columns=["embedding"]), emb_df], axis=1)

    keep_cols = ["commit_id", "project", "author_date", "buggy"] + PCA_COLS
    df = df[keep_cols].dropna(subset=["author_date"]).copy()
    df["buggy"] = df["buggy"].astype(int)

    # Verify feature integrity (Diagnostic E)
    x_all = df[PCA_COLS].to_numpy()
    nan_count = int(np.isnan(x_all).sum())
    inf_count = int(np.isinf(x_all).sum())
    const_cols = int((np.std(x_all, axis=0) == 0).sum())
    mean_feature_norm = float(np.mean(np.linalg.norm(x_all, axis=1)))

    logger.info("=== CodeBERT PCA Feature Integrity Check ===")
    logger.info("Total rows: %d, PCA features: %d", len(df), len(PCA_COLS))
    logger.info("NaN count: %d, Inf count: %d, Constant features: %d", nan_count, inf_count, const_cols)
    logger.info("Mean embedding norm: %.4f", mean_feature_norm)

    # 2. Chronological splits: 6,000 / 2,000 / 2,000
    n = len(df)
    test_start = int(n * 0.8)
    dev_df = df.iloc[:test_start].copy()
    test_df = df.iloc[test_start:].copy()

    val_start_in_dev = int(len(dev_df) * 0.75)
    train_df = dev_df.iloc[:val_start_in_dev].copy()
    val_df = dev_df.iloc[val_start_in_dev:].copy()

    # Dynamic prevalence & class counts
    train_counts = train_df["buggy"].value_counts().to_dict()
    val_counts = val_df["buggy"].value_counts().to_dict()
    test_counts = test_df["buggy"].value_counts().to_dict()

    train_neg = int(train_counts.get(0, 0))
    train_pos = int(train_counts.get(1, 0))
    val_neg = int(val_counts.get(0, 0))
    val_pos = int(val_counts.get(1, 0))
    test_neg = int(test_counts.get(0, 0))
    test_pos = int(test_counts.get(1, 0))

    train_prevalence = train_pos / len(train_df)
    val_prevalence = val_pos / len(val_df)
    test_prevalence = test_pos / len(test_df)

    dynamic_train_spw = float(train_neg / train_pos) if train_pos > 0 else 1.0
    frozen_baseline_spw = 4.908419497784343

    logger.info("=== Dataset Splits and Class Distribution ===")
    logger.info(
        "Train: %d rows (%s to %s) | Buggy: %d, Clean: %d (Prevalence: %.2f%%) | dynamic SPW=%.4f",
        len(train_df),
        train_df["author_date"].min(),
        train_df["author_date"].max(),
        train_pos,
        train_neg,
        train_prevalence * 100,
        dynamic_train_spw,
    )
    logger.info(
        "Val:   %d rows (%s to %s) | Buggy: %d, Clean: %d (Prevalence: %.2f%%)",
        len(val_df),
        val_df["author_date"].min(),
        val_df["author_date"].max(),
        val_pos,
        val_neg,
        val_prevalence * 100,
    )
    logger.info(
        "Test:  %d rows (%s to %s) | Buggy: %d, Clean: %d (Prevalence: %.2f%%)",
        len(test_df),
        test_df["author_date"].min(),
        test_df["author_date"].max(),
        test_pos,
        test_neg,
        test_prevalence * 100,
    )

    x_train, y_train = train_df[PCA_COLS], train_df["buggy"]
    x_val, y_val = val_df[PCA_COLS], val_df["buggy"].to_numpy()
    x_test, y_test = test_df[PCA_COLS], test_df["buggy"].to_numpy()

    # 3. Model configurations
    # We train:
    # A) XGBoost Unweighted (SPW=1.0)
    # B) XGBoost Dynamic SPW (SPW = train_neg / train_pos)
    # C) XGBoost Frozen SPW (SPW = 4.9084)
    # D) RandomForest Balanced (class_weight="balanced", original Exp 2 default)
    # E) RandomForest Unweighted (class_weight=None)
    model_configs = {
        "xgb_unweighted": {
            "type": "xgb",
            "spw": 1.0,
            "label": "XGBoost Unweighted (SPW=1.0)",
        },
        "xgb_dynamic_spw": {
            "type": "xgb",
            "spw": dynamic_train_spw,
            "label": f"XGBoost Dynamic SPW ({dynamic_train_spw:.4f})",
        },
        "xgb_frozen_spw": {
            "type": "xgb",
            "spw": frozen_baseline_spw,
            "label": f"XGBoost Frozen SPW ({frozen_baseline_spw:.4f})",
        },
        "rf_balanced": {
            "type": "rf",
            "class_weight": "balanced",
            "label": "RandomForest Balanced (class_weight='balanced')",
        },
        "rf_unweighted": {
            "type": "rf",
            "class_weight": None,
            "label": "RandomForest Unweighted",
        },
    }

    trained_models = {}
    val_raw_scores = {}
    test_raw_scores = {}
    platt_calibrators = {}
    val_cal_scores = {}
    test_cal_scores = {}

    for cfg_key, cfg in model_configs.items():
        if cfg["type"] == "xgb":
            logger.info("Training %s (spw=%.4f)...", cfg["label"], cfg["spw"])
            model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                learning_rate=0.1,
                n_estimators=100,
                scale_pos_weight=cfg["spw"],
                random_state=42,
            )
            model.fit(x_train, y_train)
        elif cfg["type"] == "rf":
            logger.info("Training %s...", cfg["label"])
            model = RandomForestClassifier(
                n_estimators=100,
                class_weight=cfg["class_weight"],
                random_state=42,
                n_jobs=-1,
            )
            model.fit(x_train, y_train)
        else:
            raise ValueError(f"Unknown model type {cfg['type']}")

        trained_models[cfg_key] = model

        # Predict raw probabilities
        v_raw = model.predict_proba(x_val)[:, 1]
        t_raw = model.predict_proba(x_test)[:, 1]
        val_raw_scores[cfg_key] = v_raw
        test_raw_scores[cfg_key] = t_raw

        # Platt scaling: fit strictly on validation scores and true val labels
        calibrator = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
        calibrator.fit(v_raw.reshape(-1, 1), y_val)
        platt_calibrators[cfg_key] = calibrator

        v_cal = calibrator.predict_proba(v_raw.reshape(-1, 1))[:, 1]
        t_cal = calibrator.predict_proba(t_raw.reshape(-1, 1))[:, 1]
        val_cal_scores[cfg_key] = v_cal
        test_cal_scores[cfg_key] = t_cal

    # 4. Evaluation Policies Matrix
    threshold_grid = np.linspace(0.01, 0.99, 99)

    policy_specs = [
        # XGBoost Unweighted
        {
            "id": "policy_1_xgb_unweighted_raw_05",
            "model_cfg": "xgb_unweighted",
            "calibration": "none",
            "threshold_strategy": "default_05",
            "description": "XGBoost Unweighted (SPW=1.0) + Raw @ 0.50 (Exp 2 XGB Default)",
        },
        {
            "id": "policy_2_xgb_unweighted_raw_val_opt",
            "model_cfg": "xgb_unweighted",
            "calibration": "none",
            "threshold_strategy": "val_opt_f1",
            "description": "XGBoost Unweighted (SPW=1.0) + Validation-tuned raw threshold",
        },
        {
            "id": "policy_3_xgb_unweighted_platt_val_opt",
            "model_cfg": "xgb_unweighted",
            "calibration": "platt",
            "threshold_strategy": "val_opt_f1",
            "description": "XGBoost Unweighted (SPW=1.0) + Platt calibration + Val-tuned threshold",
        },
        # XGBoost Dynamic SPW
        {
            "id": "policy_4_xgb_dynamic_spw_raw_05",
            "model_cfg": "xgb_dynamic_spw",
            "calibration": "none",
            "threshold_strategy": "default_05",
            "description": f"XGBoost Dynamic SPW ({dynamic_train_spw:.4f}) + Raw @ 0.50",
        },
        {
            "id": "policy_5_xgb_dynamic_spw_raw_val_opt",
            "model_cfg": "xgb_dynamic_spw",
            "calibration": "none",
            "threshold_strategy": "val_opt_f1",
            "description": f"XGBoost Dynamic SPW ({dynamic_train_spw:.4f}) + Val-tuned raw threshold",
        },
        {
            "id": "policy_6_xgb_dynamic_spw_platt_val_opt",
            "model_cfg": "xgb_dynamic_spw",
            "calibration": "platt",
            "threshold_strategy": "val_opt_f1",
            "description": f"XGBoost Dynamic SPW ({dynamic_train_spw:.4f}) + Platt + Val-tuned threshold",
        },
        # XGBoost Frozen SPW
        {
            "id": "policy_7_xgb_frozen_spw_raw_05",
            "model_cfg": "xgb_frozen_spw",
            "calibration": "none",
            "threshold_strategy": "default_05",
            "description": f"XGBoost Frozen SPW ({frozen_baseline_spw:.4f}) + Raw @ 0.50",
        },
        {
            "id": "policy_8_xgb_frozen_spw_raw_val_opt",
            "model_cfg": "xgb_frozen_spw",
            "calibration": "none",
            "threshold_strategy": "val_opt_f1",
            "description": f"XGBoost Frozen SPW ({frozen_baseline_spw:.4f}) + Val-tuned raw threshold",
        },
        # RandomForest Balanced (Original Exp 2 RF)
        {
            "id": "policy_9_rf_balanced_raw_05",
            "model_cfg": "rf_balanced",
            "calibration": "none",
            "threshold_strategy": "default_05",
            "description": "RandomForest Balanced + Raw @ 0.50 (Exp 2 RF Original)",
        },
        {
            "id": "policy_10_rf_balanced_raw_val_opt",
            "model_cfg": "rf_balanced",
            "calibration": "none",
            "threshold_strategy": "val_opt_f1",
            "description": "RandomForest Balanced + Val-tuned raw threshold",
        },
        {
            "id": "policy_11_rf_balanced_platt_val_opt",
            "model_cfg": "rf_balanced",
            "calibration": "platt",
            "threshold_strategy": "val_opt_f1",
            "description": "RandomForest Balanced + Platt + Val-tuned threshold",
        },
        # RandomForest Unweighted
        {
            "id": "policy_12_rf_unweighted_raw_05",
            "model_cfg": "rf_unweighted",
            "calibration": "none",
            "threshold_strategy": "default_05",
            "description": "RandomForest Unweighted + Raw @ 0.50",
        },
        {
            "id": "policy_13_rf_unweighted_raw_val_opt",
            "model_cfg": "rf_unweighted",
            "calibration": "none",
            "threshold_strategy": "val_opt_f1",
            "description": "RandomForest Unweighted + Val-tuned raw threshold",
        },
    ]

    val_policy_rows = []
    policy_artifacts = {}

    for spec in policy_specs:
        cfg_key = spec["model_cfg"]
        is_cal = spec["calibration"] == "platt"

        val_scores = val_cal_scores[cfg_key] if is_cal else val_raw_scores[cfg_key]
        test_scores = test_cal_scores[cfg_key] if is_cal else test_raw_scores[cfg_key]

        if spec["threshold_strategy"] == "default_05":
            chosen_threshold = 0.50
            val_metrics = compute_metrics_dict(y_val, val_scores, chosen_threshold)
            sweep_table = None
        else:
            chosen_threshold, val_metrics, sweep_table = find_optimal_threshold(
                y_val, val_scores, threshold_grid, metric_key="f1"
            )

        val_row = {
            "policy_id": spec["id"],
            "description": spec["description"],
            "model": cfg_key,
            "calibration": spec["calibration"],
            "threshold": float(round(chosen_threshold, 4)),
            "accuracy": val_metrics["accuracy"],
            "precision": val_metrics["precision"],
            "recall": val_metrics["recall"],
            "f1": val_metrics["f1"],
            "mcc": val_metrics["mcc"],
            "pr_auc": val_metrics["pr_auc"],
            "roc_auc": val_metrics["roc_auc"],
            "brier": val_metrics["brier"],
            "ece": val_metrics["ece"],
            "predicted_positive_rate": val_metrics["predicted_positive_rate"],
            "tn": val_metrics["tn"],
            "fp": val_metrics["fp"],
            "fn": val_metrics["fn"],
            "tp": val_metrics["tp"],
        }
        val_policy_rows.append(val_row)

        policy_artifacts[spec["id"]] = {
            "spec": spec,
            "chosen_threshold": chosen_threshold,
            "val_metrics": val_metrics,
            "sweep_table": sweep_table,
            "val_scores": val_scores,
            "test_scores": test_scores,
        }

    val_comparison_df = pd.DataFrame(val_policy_rows)
    val_comparison_df.to_csv(OUTPUT_DIR / "tables" / "validation_policies_comparison.csv", index=False)

    logger.info("=== Validation Policy Comparison Table ===")
    print(val_comparison_df[["policy_id", "threshold", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc", "predicted_positive_rate"]].to_string(index=False))

    # 5. Evaluate all policies on the untouched Final Test period
    test_policy_rows = []
    test_predictions_export = test_df[["commit_id", "project", "author_date"]].copy()
    test_predictions_export["buggy_true"] = y_test

    for spec in policy_specs:
        p_id = spec["id"]
        art = policy_artifacts[p_id]
        chosen_threshold = art["chosen_threshold"]
        test_scores = art["test_scores"]

        test_metrics = compute_metrics_dict(y_test, test_scores, chosen_threshold)
        art["test_metrics"] = test_metrics

        test_row = {
            "policy_id": p_id,
            "description": spec["description"],
            "model": spec["model_cfg"],
            "calibration": spec["calibration"],
            "threshold": float(round(chosen_threshold, 4)),
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1": test_metrics["f1"],
            "mcc": test_metrics["mcc"],
            "pr_auc": test_metrics["pr_auc"],
            "roc_auc": test_metrics["roc_auc"],
            "brier": test_metrics["brier"],
            "ece": test_metrics["ece"],
            "predicted_positive_rate": test_metrics["predicted_positive_rate"],
            "tn": test_metrics["tn"],
            "fp": test_metrics["fp"],
            "fn": test_metrics["fn"],
            "tp": test_metrics["tp"],
        }
        test_policy_rows.append(test_row)

        test_predictions_export[f"score_{p_id}"] = test_scores
        test_predictions_export[f"pred_{p_id}"] = (test_scores >= chosen_threshold).astype(int)

    test_comparison_df = pd.DataFrame(test_policy_rows)
    test_comparison_df.to_csv(OUTPUT_DIR / "tables" / "final_test_policies_comparison.csv", index=False)
    test_predictions_export.to_csv(OUTPUT_DIR / "predictions" / "final_test_predictions.csv", index=False)

    logger.info("=== Final Test Policy Comparison Table ===")
    print(test_comparison_df[["policy_id", "threshold", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc", "predicted_positive_rate"]].to_string(index=False))

    # 6. Diagnostic Visualizations
    diag_dict = {
        "XGB Unweighted @ 0.5": {
            "score": policy_artifacts["policy_1_xgb_unweighted_raw_05"]["test_scores"],
            "threshold": 0.50,
        },
        "XGB Dynamic SPW @ 0.5": {
            "score": policy_artifacts["policy_4_xgb_dynamic_spw_raw_05"]["test_scores"],
            "threshold": 0.50,
        },
        "XGB Dynamic SPW (Val-Tuned)": {
            "score": policy_artifacts["policy_5_xgb_dynamic_spw_raw_val_opt"]["test_scores"],
            "threshold": policy_artifacts["policy_5_xgb_dynamic_spw_raw_val_opt"]["chosen_threshold"],
        },
        "XGB Dynamic SPW (Platt)": {
            "score": policy_artifacts["policy_6_xgb_dynamic_spw_platt_val_opt"]["test_scores"],
            "threshold": policy_artifacts["policy_6_xgb_dynamic_spw_platt_val_opt"]["chosen_threshold"],
        },
        "RF Balanced @ 0.5 (Original Exp 2)": {
            "score": policy_artifacts["policy_9_rf_balanced_raw_05"]["test_scores"],
            "threshold": 0.50,
        },
    }
    plot_diagnostics(y_test, diag_dict, OUTPUT_DIR)

    # 7. Comparison with Experiment 1 Controlled Baseline
    exp1_summary = None
    if EXP1_CONTROLLED_SUMMARY.exists():
        exp1_summary = json.loads(EXP1_CONTROLLED_SUMMARY.read_text(encoding="utf-8"))

    # Select primary CodeBERT policy based on validation evidence
    # We examine the best validation F1 / MCC policy
    best_val_policy = val_comparison_df.sort_values(["f1", "mcc", "precision"], ascending=[False, False, False]).iloc[0]
    selected_policy_id = str(best_val_policy["policy_id"])
    selected_art = policy_artifacts[selected_policy_id]

    summary_data = {
        "experiment": "java_only_experiment_2_controlled_v1",
        "methodology": "Train (6000) -> Temporal Validation (2000) -> Untouched Final Test (2000)",
        "features": {
            "type": "CodeBERT PCA (384 components)",
            "nan_count": nan_count,
            "inf_count": inf_count,
            "constant_features": const_cols,
            "mean_feature_norm": mean_feature_norm,
        },
        "split_summary": {
            "total_rows": int(len(df)),
            "train": {
                "rows": int(len(train_df)),
                "clean_count": train_neg,
                "buggy_count": train_pos,
                "buggy_prevalence_pct": float(round(train_prevalence * 100, 3)),
                "start_date": str(train_df["author_date"].min()),
                "end_date": str(train_df["author_date"].max()),
                "dynamic_scale_pos_weight": dynamic_train_spw,
            },
            "validation": {
                "rows": int(len(val_df)),
                "clean_count": val_neg,
                "buggy_count": val_pos,
                "buggy_prevalence_pct": float(round(val_prevalence * 100, 3)),
                "start_date": str(val_df["author_date"].min()),
                "end_date": str(val_df["author_date"].max()),
            },
            "final_test": {
                "rows": int(len(test_df)),
                "clean_count": test_neg,
                "buggy_count": test_pos,
                "buggy_prevalence_pct": float(round(test_prevalence * 100, 3)),
                "start_date": str(test_df["author_date"].min()),
                "end_date": str(test_df["author_date"].max()),
            },
        },
        "validation_policies": val_policy_rows,
        "selected_policy": {
            "policy_id": selected_policy_id,
            "description": selected_art["spec"]["description"],
            "rationale": f"Top validation F1 ({best_val_policy['f1']:.4f}) and MCC ({best_val_policy['mcc']:.4f})",
            "locked_parameters": {
                "model": selected_art["spec"]["model_cfg"],
                "calibration": selected_art["spec"]["calibration"],
                "threshold": float(round(selected_art["chosen_threshold"], 4)),
            },
            "validation_metrics": selected_art["val_metrics"],
            "final_test_metrics": selected_art["test_metrics"],
        },
        "all_final_test_policies": test_policy_rows,
        "exp1_controlled_jit_baseline": exp1_summary.get("selected_policy") if exp1_summary else None,
    }

    summary_path = OUTPUT_DIR / "metrics" / "summary.json"
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    logger.info("Saved experiment summary to %s", summary_path)

    return summary_data


if __name__ == "__main__":
    run_experiment()
