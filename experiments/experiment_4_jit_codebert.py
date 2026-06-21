"""Experiment 4: JIT + CodeBERT feature fusion.

This script loads the frozen sample dataset from `results/frozen_sample_dataset.csv`,
selects traditional JIT metrics and PCA-reduced CodeBERT features, performs a
chronological temporal split, trains Random Forest and XGBoost models, and
saves metrics, predictions, models, and visualizations.

No additional sampling, PCA, or embedding regeneration is performed.
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

INPUT_PATH = PROJECT_ROOT / "results" / "frozen_sample_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "experiment_4"
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
FEATURE_COLUMNS = JIT_FEATURES + PCA_FEATURES


def ensure_directories() -> None:
    for path in [METRICS_DIR, PREDICTIONS_DIR, PLOTS_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, object], path: Path) -> None:
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
                try:
                    return json.loads(value)
                except Exception as exc:
                    raise ValueError(f"Could not parse embedding string: {exc}") from exc
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


def plot_roc_pr(y_true: Sequence[int], scores: Dict[str, Sequence[float]], output_path: Path) -> None:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for label, y_score in scores.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc(fpr, tpr):.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for label, y_score in scores.items():
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        plt.plot(recall, precision, label=f"{label}")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.grid(True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info("Saved ROC/PR plot to %s", output_path)


def plot_feature_importance(model: object, feature_names: List[str], output_path: Path, title: str) -> None:
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model %s does not have feature_importances_", model.__class__.__name__)
        return

    importance_df = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
    importance_df = importance_df.sort_values(by="importance", ascending=True)

    plt.figure(figsize=(10, 14))
    plt.barh(importance_df["feature"], importance_df["importance"], color="#5A7FA8")
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info("Saved feature importance plot to %s", output_path)


def plot_model_comparison(metrics: Dict[str, Dict[str, float]], output_path: Path) -> None:
    labels = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc"]
    x = np.arange(len(labels))
    width = 0.35

    rf_values = [metrics["random_forest"][label] for label in labels]
    xgb_values = [metrics["xgboost"][label] for label in labels]

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, rf_values, width, label="Random Forest")
    plt.bar(x + width / 2, xgb_values, width, label="XGBoost")
    plt.xticks(x, labels, rotation=45)
    plt.ylabel("Score")
    plt.title("Model Comparison")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info("Saved model comparison plot to %s", output_path)


def save_feature_importance_csv(model: object, feature_names: List[str], path: Path) -> None:
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model %s does not have feature_importances_", model.__class__.__name__)
        return

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values(by="importance", ascending=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(path, index=False)
    logger.info("Saved feature importance CSV to %s", path)


def main() -> None:
    logger.info("Experiment 4 (JIT + CodeBERT) starting")
    ensure_directories()

    if not INPUT_PATH.exists():
        logger.error("Frozen sample not found at %s", INPUT_PATH)
        raise FileNotFoundError(f"Frozen sample not found: {INPUT_PATH}")

    df = load_commits_csv(INPUT_PATH)
    logger.info("Loaded frozen sample: %d rows, %d columns", len(df), len(df.columns))

    df = expand_embedding_to_pca(df, PCA_FEATURES)

    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        logger.error("Required feature columns missing: %s", missing_features[:10])
        raise KeyError(f"Missing feature columns: {missing_features}")

    keep_cols = ["commit_id", "project", "author_date", "buggy"] + FEATURE_COLUMNS
    df_reduced = df[keep_cols].copy()
    df_reduced = df_reduced.sort_values("author_date").reset_index(drop=True)

    n_missing = int(df_reduced[FEATURE_COLUMNS].isna().any(axis=1).sum())
    if n_missing > 0:
        logger.warning("Dropping %d rows with missing feature values", n_missing)
        df_reduced = df_reduced.dropna(subset=FEATURE_COLUMNS)

    X_train, X_test, y_train, y_test = temporal_train_test_split(
        df_reduced,
        timestamp_column="author_date",
        target_column="buggy",
        feature_columns=FEATURE_COLUMNS,
        train_size=0.8,
    )

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    scale_pos_weight = 1.0
    if y_train.sum() > 0:
        scale_pos_weight = float((len(y_train) - y_train.sum()) / y_train.sum())
    logger.info("Computed scale_pos_weight=%s from training label distribution", scale_pos_weight)

    rf_model_path = MODELS_DIR / "random_forest_experiment_4.joblib"
    xgb_model_path = MODELS_DIR / "xgboost_experiment_4.joblib"

    logger.info("Training Random Forest")
    rf_model, rf_pred, rf_score = train_random_forest(
        X_train,
        y_train,
        X_test,
        y_test,
        model_path=rf_model_path,
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )
    rf_score = ensure_prob_array(rf_score)
    rf_metrics = compute_classification_metrics(y_test, rf_pred, rf_score)
    save_json(rf_metrics, METRICS_DIR / "random_forest_metrics.json")

    logger.info("Training XGBoost")
    xgb_model, xgb_pred, xgb_score = train_xgboost(
        X_train,
        y_train,
        X_test,
        y_test,
        model_path=xgb_model_path,
        n_estimators=100,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )
    xgb_score = ensure_prob_array(xgb_score)
    xgb_metrics = compute_classification_metrics(y_test, xgb_pred, xgb_score)
    save_json(xgb_metrics, METRICS_DIR / "xgboost_metrics.json")

    summary = {
        "experiment": "experiment_4_jit_codebert",
        "dataset_rows": len(df_reduced),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "jit_features": JIT_FEATURES,
        "codebert_pca_features": "pca_1..pca_384",
        "models": {
            "random_forest": rf_metrics,
            "xgboost": xgb_metrics,
        },
        "scale_pos_weight": scale_pos_weight,
    }
    save_json(summary, METRICS_DIR / "summary.json")

    predictions_meta = df_reduced.loc[X_test.index, ["commit_id", "project", "author_date"]].copy()
    predictions_df = predictions_meta.assign(
        buggy_true=y_test.values,
        rf_pred=rf_pred.values,
        rf_score=rf_score,
        xgb_pred=xgb_pred.values,
        xgb_score=xgb_score,
    )
    predictions_path = PREDICTIONS_DIR / "experiment_4_jit_codebert_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    logger.info("Saved predictions to %s", predictions_path)

    save_feature_importance_csv(
        rf_model,
        FEATURE_COLUMNS,
        METRICS_DIR / "experiment_4_rf_feature_importance.csv",
    )
    save_feature_importance_csv(
        xgb_model,
        FEATURE_COLUMNS,
        METRICS_DIR / "experiment_4_xgb_feature_importance.csv",
    )

    plot_roc_pr(
        y_test,
        {
            "Random Forest": rf_score,
            "XGBoost": xgb_score,
        },
        PLOTS_DIR / "experiment_4_jit_codebert_roc_pr.png",
    )

    plot_feature_importance(
        rf_model,
        FEATURE_COLUMNS,
        PLOTS_DIR / "experiment_4_jit_codebert_rf_feature_importance.png",
        title="Random Forest Feature Importance",
    )
    plot_feature_importance(
        xgb_model,
        FEATURE_COLUMNS,
        PLOTS_DIR / "experiment_4_jit_codebert_xgb_feature_importance.png",
        title="XGBoost Feature Importance",
    )

    plot_model_comparison(
        {"random_forest": rf_metrics, "xgboost": xgb_metrics},
        PLOTS_DIR / "experiment_4_jit_codebert_model_comparison.png",
    )

    logger.info("Experiment 4 completed successfully")


if __name__ == "__main__":
    main()
