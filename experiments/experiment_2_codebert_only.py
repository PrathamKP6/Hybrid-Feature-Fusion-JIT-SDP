"""Experiment 2: CodeBERT features only.

Loads the frozen sample dataset and evaluates models using only the
CodeBERT PCA components (pca_1 .. pca_384). Performs a chronological
temporal split (first 80% train, last 20% test), trains RandomForest
and XGBoost models, evaluates metrics, saves models, predictions,
metrics, and basic ROC/PR plots.

Requirements:
- Uses `results/frozen_sample_dataset.csv` as input
- Reuses existing training and metric utilities
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.train_random_forest import train_random_forest
from models.train_xgboost import train_xgboost
from preprocessing.temporal_split import temporal_train_test_split
from utils.metrics import compute_classification_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

INPUT_PATH = PROJECT_ROOT / "results" / "data" / "frozen_sample_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "experiment_2"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODELS_DIR = OUTPUT_DIR / "models"
PRED_DIR = OUTPUT_DIR / "predictions"
PLOTS_DIR = OUTPUT_DIR / "plots"


def pca_columns(start: int = 1, end: int = 384) -> list[str]:
    return [f"pca_{i}" for i in range(start, end + 1)]


def ensure_prob_array(y_score: np.ndarray) -> np.ndarray:
    arr = np.asarray(y_score)
    if arr.ndim == 2 and arr.shape[1] > 1:
        return arr[:, 1]
    return arr.ravel()


def plot_roc_pr(y_true: Sequence[int], y_score: Sequence[float], out_prefix: Path) -> None:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    roc_path = out_prefix.parent / (out_prefix.name + "_roc.png")
    roc_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(roc_path), dpi=150)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f"AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    pr_path = out_prefix.parent / (out_prefix.name + "_pr.png")
    pr_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(pr_path), dpi=150)
    plt.close()


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def main() -> None:
    logger.info("Experiment 2 (CodeBERT only) starting")

    if not INPUT_PATH.exists():
        logger.error("Frozen sample not found at %s", INPUT_PATH)
        raise FileNotFoundError(f"Frozen sample not found: {INPUT_PATH}")

    import ast

    df = pd.read_csv(INPUT_PATH, parse_dates=["author_date"], low_memory=False)
    logger.info("Loaded frozen sample: %d rows, %d columns", len(df), len(df.columns))

    pca_cols = pca_columns()
    missing = [c for c in pca_cols if c not in df.columns]
    if missing:
        # Attempt to expand stored `embedding` object into pca columns
        if "embedding" in df.columns:
            logger.info("No pca_ columns found — expanding 'embedding' into %d columns", len(pca_cols))
            # parse embedding strings if necessary
            def _parse_emb(x):
                if pd.isna(x):
                    return [0.0] * len(pca_cols)
                if isinstance(x, str):
                    try:
                        return ast.literal_eval(x)
                    except Exception:
                        try:
                            return json.loads(x)
                        except Exception:
                            raise ValueError("Could not parse embedding string")
                if isinstance(x, (list, tuple, np.ndarray)):
                    return list(x)
                raise ValueError("Unknown embedding format")

            emb_series = df["embedding"].apply(_parse_emb)
            emb_matrix = np.vstack(emb_series.values)
            if emb_matrix.shape[1] != len(pca_cols):
                raise ValueError(f"Embedding dimension {emb_matrix.shape[1]} does not match expected {len(pca_cols)}")
            emb_df = pd.DataFrame(emb_matrix, columns=pca_cols, index=df.index)
            df = pd.concat([df.drop(columns=["embedding"]), emb_df], axis=1)
            missing = [c for c in pca_cols if c not in df.columns]
            if missing:
                logger.error("After expansion still missing columns: %s", missing[:10])
                raise KeyError(f"Missing PCA columns after expansion: {missing}")
            logger.info("Expanded embedding into pca columns successfully")
        else:
            logger.error("Missing PCA columns: %s", missing[:10])
            raise KeyError(f"Missing PCA columns: {missing}")

    # Keep only PCA features + id, timestamp, target for bookkeeping
    keep_cols = ["commit_id", "author_date", "project", "buggy"] + pca_cols
    df_reduced = df[keep_cols].copy()
    df_reduced = df_reduced.sort_values("author_date")

    X_train, X_test, y_train, y_test = temporal_train_test_split(
        df_reduced,
        timestamp_column="author_date",
        target_column="buggy",
        feature_columns=pca_cols,
        train_size=0.8,
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Random Forest
    rf_model_path = MODELS_DIR / "random_forest_codebert.joblib"
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
    rf_metrics = compute_classification_metrics(y_test, rf_pred, rf_score, pos_label=1)
    save_json(METRICS_DIR / "random_forest_metrics.json", rf_metrics)

    rf_pred_df = pd.DataFrame({
        "commit_id": df_reduced.iloc[X_test.index]["commit_id"].values if hasattr(X_test, "index") else df_reduced["commit_id"].iloc[-len(rf_pred):].values,
        "author_date": df_reduced.iloc[X_test.index]["author_date"].values if hasattr(X_test, "index") else None,
        "y_true": y_test,
        "y_pred": rf_pred,
        "y_score": rf_score,
    })
    rf_pred_df.to_csv(PRED_DIR / "random_forest_predictions.csv", index=False)
    plot_roc_pr(y_test, rf_score, PLOTS_DIR / "random_forest")

    logger.info("Random Forest metrics: %s", {k: rf_metrics[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")})

    # XGBoost
    xgb_model_path = MODELS_DIR / "xgboost_codebert.joblib"
    logger.info("Training XGBoost")
    xgb_model, xgb_pred, xgb_score = train_xgboost(
        X_train,
        y_train,
        X_test,
        y_test,
        model_path=xgb_model_path,
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
    )
    xgb_score = ensure_prob_array(xgb_score)
    xgb_metrics = compute_classification_metrics(y_test, xgb_pred, xgb_score, pos_label=1)
    save_json(METRICS_DIR / "xgboost_metrics.json", xgb_metrics)

    xgb_pred_df = pd.DataFrame({
        "commit_id": df_reduced.iloc[X_test.index]["commit_id"].values if hasattr(X_test, "index") else df_reduced["commit_id"].iloc[-len(xgb_pred):].values,
        "author_date": df_reduced.iloc[X_test.index]["author_date"].values if hasattr(X_test, "index") else None,
        "y_true": y_test,
        "y_pred": xgb_pred,
        "y_score": xgb_score,
    })
    xgb_pred_df.to_csv(PRED_DIR / "xgboost_predictions.csv", index=False)
    plot_roc_pr(y_test, xgb_score, PLOTS_DIR / "xgboost")

    logger.info("XGBoost metrics: %s", {k: xgb_metrics[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")})

    # Save combined summary
    summary = {"random_forest": rf_metrics, "xgboost": xgb_metrics}
    save_json(METRICS_DIR / "summary.json", summary)

    logger.info("Experiment 2 complete. Results saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
