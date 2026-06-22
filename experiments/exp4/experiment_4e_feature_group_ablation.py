"""Experiment 5: Feature group ablation study.

This script compares JIT-only, CodeBERT PCA-only, and JIT+CodeBERT fusion
configurations using Random Forest and XGBoost. It uses the frozen sample
without additional sampling, PCA generation, embedding generation, or data
balancing outside model-level weighting.
"""
from __future__ import annotations

import ast
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.train_random_forest import train_random_forest
from models.train_xgboost import train_xgboost
from preprocessing.load_data import load_commits_csv
from preprocessing.temporal_split import temporal_train_test_split
from utils.metrics import compute_classification_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

INPUT_PATH = PROJECT_ROOT / "results" / "data" / "frozen_sample_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "exp4" / "experiment_4e"
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
PLOTS_DIR = OUTPUT_DIR / "plots"
MODELS_DIR = OUTPUT_DIR / "models"

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


def pca_columns(start: int = 1, end: int = 384) -> List[str]:
    return [f"pca_{i}" for i in range(start, end + 1)]

PCA_FEATURES = pca_columns()
FEATURE_SETS = {
    "jit_only": JIT_FEATURES,
    "pca_only": PCA_FEATURES,
    "fusion": JIT_FEATURES + PCA_FEATURES,
}


def ensure_directories() -> None:
    for path in [METRICS_DIR, PREDICTIONS_DIR, PLOTS_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    logger.info("Saved JSON to %s", path)


def ensure_prob_array(y_score: Sequence[float]) -> np.ndarray:
    arr = np.asarray(y_score)
    if arr.ndim == 2 and arr.shape[1] > 1:
        return arr[:, 1]
    return arr.ravel()


def expand_embedding_to_pca(df: pd.DataFrame, pca_cols: List[str]) -> pd.DataFrame:
    missing = [c for c in pca_cols if c not in df.columns]
    if not missing:
        return df

    if "embedding" not in df.columns:
        logger.error("Missing PCA columns and no 'embedding' column available")
        raise KeyError(f"Missing PCA columns: {missing}")

    logger.info("Expanding 'embedding' column to %d PCA columns", len(pca_cols))

    def _parse_embedding(value: object) -> list[float]:
        if pd.isna(value):
            return [0.0] * len(pca_cols)
        if isinstance(value, (list, tuple, np.ndarray)):
            return list(value)
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)
            except Exception:
                return json.loads(value)
        raise ValueError(f"Unsupported embedding format: {type(value)}")

    emb_series = df["embedding"].apply(_parse_embedding)
    emb_matrix = np.vstack(emb_series.values)
    if emb_matrix.shape[1] != len(pca_cols):
        raise ValueError(
            f"Embedding dimension {emb_matrix.shape[1]} does not match expected {len(pca_cols)}"
        )

    emb_df = pd.DataFrame(emb_matrix, columns=pca_cols, index=df.index)
    df = pd.concat([df.drop(columns=["embedding"]), emb_df], axis=1)
    logger.info("Expanded embeddings into PCA columns successfully")
    return df


def plot_roc_pr_comparison(scores: Dict[str, Sequence[float]], y_true: Sequence[int], output_path: Path) -> None:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for label, y_score in scores.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, label=f"{label} ({auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right", fontsize="small")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for label, y_score in scores.items():
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        plt.plot(recall, precision, label=label)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Comparison")
    plt.legend(loc="lower left", fontsize="small")
    plt.grid(True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info("Saved ROC/PR comparison plot to %s", output_path)


def plot_metric_bars(metrics: Dict[str, Dict[str, float]], output_path: Path) -> None:
    labels = ["roc_auc", "pr_auc", "f1", "mcc"]
    configs = list(metrics.keys())
    x = np.arange(len(labels))
    width = 0.13

    plt.figure(figsize=(14, 7))
    for idx, config in enumerate(configs):
        values = [metrics[config][label] for label in labels]
        plt.bar(x + (idx - len(configs) / 2) * width + width / 2, values, width, label=config)

    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.title("Ablation Metrics Comparison")
    plt.ylim(0, 1)
    plt.legend(loc="upper left", fontsize="small", ncol=2)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info("Saved ablation metric comparison plot to %s", output_path)


def plot_feature_importance(model: object, feature_names: List[str], output_path: Path, title: str) -> None:
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model %s does not have feature_importances_", model.__class__.__name__)
        return

    importance_df = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
    importance_df = importance_df.sort_values(by="importance", ascending=True)

    plt.figure(figsize=(10, 14))
    plt.barh(importance_df["feature"], importance_df["importance"], color="#4C72B0")
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info("Saved feature importance plot to %s", output_path)


def save_feature_importance_csv(model: object, feature_names: List[str], path: Path) -> None:
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model %s does not have feature_importances_", model.__class__.__name__)
        return

    importance_df = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
    importance_df = importance_df.sort_values(by="importance", ascending=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(path, index=False)
    logger.info("Saved feature importance CSV to %s", path)


def interpret_gain(gain: float) -> str:
    if gain > 0.05:
        return "significant improvement"
    if gain > 0.01:
        return "marginal improvement"
    if gain >= -0.01:
        return "no improvement"
    return "performance degradation"


def train_and_evaluate(
    df: pd.DataFrame,
    feature_columns: List[str],
    config_label: str,
    model_type: str,
) -> Dict[str, object]:
    X_train, X_test, y_train, y_test = temporal_train_test_split(
        df,
        timestamp_column="author_date",
        target_column="buggy",
        feature_columns=feature_columns,
        train_size=0.8,
    )

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    scale_pos_weight = 1.0
    if y_train.sum() > 0:
        scale_pos_weight = float((len(y_train) - y_train.sum()) / y_train.sum())

    model_path = MODELS_DIR / f"{model_type}_{config_label}.joblib"
    if model_type == "rf":
        model, y_pred, y_score = train_random_forest(
            X_train,
            y_train,
            X_test,
            y_test,
            model_path=model_path,
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
        )
    else:
        model, y_pred, y_score = train_xgboost(
            X_train,
            y_train,
            X_test,
            y_test,
            model_path=model_path,
            n_estimators=100,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
        )

    y_score = ensure_prob_array(y_score)
    metrics = compute_classification_metrics(y_test, y_pred, y_score)

    save_json(METRICS_DIR / f"{model_type}_{config_label}_metrics.json", metrics)

    predictions_meta = df.loc[X_test.index, ["commit_id", "project", "author_date"]].copy()
    predictions_df = predictions_meta.assign(
        buggy_true=y_test.values,
        y_pred=y_pred.values,
        y_score=y_score,
    )
    predictions_df.to_csv(PREDICTIONS_DIR / f"{model_type}_{config_label}_predictions.csv", index=False)

    return {
        "model": model,
        "metrics": metrics,
        "y_score": y_score,
        "y_test": y_test,
        "feature_columns": feature_columns,
    }


def main() -> None:
    logger.info("Experiment 5 (feature group ablation) starting")
    ensure_directories()

    if not INPUT_PATH.exists():
        logger.error("Frozen sample not found at %s", INPUT_PATH)
        raise FileNotFoundError(f"Frozen sample not found: {INPUT_PATH}")

    df = load_commits_csv(INPUT_PATH)
    logger.info("Loaded frozen sample: %d rows, %d columns", len(df), len(df.columns))

    df = expand_embedding_to_pca(df, PCA_FEATURES)

    required_cols = ["commit_id", "project", "author_date", "buggy"] + JIT_FEATURES + PCA_FEATURES
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        logger.error("Missing required columns: %s", missing[:20])
        raise KeyError(f"Missing required columns: {missing}")

    df_reduced = df[required_cols].copy()
    df_reduced = df_reduced.sort_values("author_date").reset_index(drop=True)

    if df_reduced[FEATURE_SETS["fusion"]].isna().any().any():
        missing_rows = int(df_reduced[FEATURE_SETS["fusion"]].isna().any(axis=1).sum())
        logger.warning("Dropping %d rows with missing feature values", missing_rows)
        df_reduced = df_reduced.dropna(subset=FEATURE_SETS["fusion"]).reset_index(drop=True)

    experiments: Dict[str, Dict[str, object]] = {}
    scores_for_plot: Dict[str, Sequence[float]] = {}
    plots_y_true = None

    for feature_label, features in FEATURE_SETS.items():
        for model_type in ["rf", "xgb"]:
            config_name = f"{model_type}_{feature_label}"
            logger.info("Running configuration %s", config_name)
            result = train_and_evaluate(df_reduced, features, feature_label, model_type)
            experiments[config_name] = result
            scores_for_plot[config_name] = result["y_score"]
            if plots_y_true is None:
                plots_y_true = result["y_test"]

    if plots_y_true is None:
        raise RuntimeError("No experiment results were generated")

    plot_roc_pr_comparison(scores_for_plot, plots_y_true, PLOTS_DIR / "experiment_4e_roc_pr_comparison.png")
    plot_metric_bars({k: v["metrics"] for k, v in experiments.items()}, PLOTS_DIR / "experiment_4e_metric_comparison.png")

    fusion_rf = experiments["rf_fusion"]["model"]
    fusion_xgb = experiments["xgb_fusion"]["model"]
    save_feature_importance_csv(
        fusion_rf,
        FEATURE_SETS["fusion"],
        METRICS_DIR / "experiment_4e_rf_fusion_feature_importance.csv",
    )
    save_feature_importance_csv(
        fusion_xgb,
        FEATURE_SETS["fusion"],
        METRICS_DIR / "experiment_4e_xgb_fusion_feature_importance.csv",
    )
    plot_feature_importance(
        fusion_rf,
        FEATURE_SETS["fusion"],
        PLOTS_DIR / "experiment_4e_rf_fusion_feature_importance.png",
        title="RF Fusion Feature Importance",
    )
    plot_feature_importance(
        fusion_xgb,
        FEATURE_SETS["fusion"],
        PLOTS_DIR / "experiment_4e_xgb_fusion_feature_importance.png",
        title="XGB Fusion Feature Importance",
    )

    jit_roc = experiments["rf_jit_only"]["metrics"]["roc_auc"] if experiments["rf_jit_only"]["metrics"]["roc_auc"] >= experiments["xgb_jit_only"]["metrics"]["roc_auc"] else experiments["xgb_jit_only"]["metrics"]["roc_auc"]
    pca_roc = experiments["rf_pca_only"]["metrics"]["roc_auc"] if experiments["rf_pca_only"]["metrics"]["roc_auc"] >= experiments["xgb_pca_only"]["metrics"]["roc_auc"] else experiments["xgb_pca_only"]["metrics"]["roc_auc"]
    fusion_roc = experiments["rf_fusion"]["metrics"]["roc_auc"] if experiments["rf_fusion"]["metrics"]["roc_auc"] >= experiments["xgb_fusion"]["metrics"]["roc_auc"] else experiments["xgb_fusion"]["metrics"]["roc_auc"]
    jit_pr = experiments["rf_jit_only"]["metrics"]["pr_auc"] if experiments["rf_jit_only"]["metrics"]["pr_auc"] >= experiments["xgb_jit_only"]["metrics"]["pr_auc"] else experiments["xgb_jit_only"]["metrics"]["pr_auc"]
    pca_pr = experiments["rf_pca_only"]["metrics"]["pr_auc"] if experiments["rf_pca_only"]["metrics"]["pr_auc"] >= experiments["xgb_pca_only"]["metrics"]["pr_auc"] else experiments["xgb_pca_only"]["metrics"]["pr_auc"]
    fusion_pr = experiments["rf_fusion"]["metrics"]["pr_auc"] if experiments["rf_fusion"]["metrics"]["pr_auc"] >= experiments["xgb_fusion"]["metrics"]["pr_auc"] else experiments["xgb_fusion"]["metrics"]["pr_auc"]

    best_config = max(experiments.items(), key=lambda item: item[1]["metrics"]["roc_auc"])[0]
    _, best_model, best_feature_set = best_config.split("_", 2)

    ablation_summary = {
        "best_model": best_model.upper(),
        "best_feature_set": best_feature_set.upper(),
        "jit_only_roc_auc": float(jit_roc),
        "pca_only_roc_auc": float(pca_roc),
        "fusion_roc_auc": float(fusion_roc),
        "jit_only_pr_auc": float(jit_pr),
        "pca_only_pr_auc": float(pca_pr),
        "fusion_pr_auc": float(fusion_pr),
        "fusion_gain_over_jit": float(fusion_roc - jit_roc),
        "fusion_gain_over_pca": float(fusion_roc - pca_roc),
        "fusion_gain_over_jit_interpretation": interpret_gain(float(fusion_roc - jit_roc)),
        "fusion_gain_over_pca_interpretation": interpret_gain(float(fusion_roc - pca_roc)),
    }
    save_json(METRICS_DIR / "ablation_summary.json", ablation_summary)

    logger.info("Experiment 5 completed successfully")


if __name__ == "__main__":
    main()
