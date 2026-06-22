"""Experiment 7: Late Fusion / Stacking Ensemble.

This script evaluates whether semantic CodeBERT PCA features provide
complementary information to JIT metrics through model-level fusion.

It trains:
- JIT XGBoost using only JIT metrics.
- Semantic XGBoost using the first 25 PCA components.
- Early Fusion XGBoost using JIT metrics + all 384 PCA features.
- Top-25 Fusion XGBoost using JIT metrics + first 25 PCA features.
- Stacking ensemble with a Logistic Regression meta-learner that uses
  JIT and semantic prediction probabilities.

The experiment saves comparison metrics, ROC/PR charts, and a best model summary.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.train_xgboost import train_xgboost
from preprocessing.load_data import load_commits_csv
from preprocessing.temporal_split import temporal_train_test_split
from utils.metrics import compute_classification_metrics

INPUT_PATH = PROJECT_ROOT / "results" / "data" / "frozen_sample_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "experiment_7"
METRICS_DIR = OUTPUT_DIR / "metrics"
PLOTS_DIR = OUTPUT_DIR / "plots"
MODELS_DIR = OUTPUT_DIR / "models"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

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

PCA_FEATURES_25 = pca_columns(1, 25)
PCA_FEATURES_384 = pca_columns(1, 384)


def ensure_directories() -> None:
    for path in [METRICS_DIR, PLOTS_DIR, MODELS_DIR, PREDICTIONS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


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
        raise KeyError(f"Missing PCA columns and no 'embedding' column available: {missing}")

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
    return pd.concat([df.drop(columns=["embedding"]), emb_df], axis=1)


def plot_roc_comparison(y_true: Sequence[int], scores: Dict[str, Sequence[float]], output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    for label, y_score in scores.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc(fpr, tpr):.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(fontsize="small")
    plt.grid(True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_pr_comparison(y_true: Sequence[int], scores: Dict[str, Sequence[float]], output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    for label, y_score in scores.items():
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        plt.plot(recall, precision, label=f"{label}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Comparison")
    plt.legend(fontsize="small")
    plt.grid(True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def train_xgb_experiment(
    label: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_columns: List[str],
) -> Dict[str, object]:
    model_path = MODELS_DIR / f"{label}.joblib"
    model, y_pred, y_score = train_xgboost(
        X_train[feature_columns],
        y_train,
        X_test[feature_columns],
        y_test,
        model_path=model_path,
        n_estimators=100,
        learning_rate=0.1,
        scale_pos_weight=float((len(y_train) - y_train.sum()) / y_train.sum()) if y_train.sum() > 0 else 1.0,
        random_state=42,
    )
    y_score = ensure_prob_array(y_score)
    metrics = compute_classification_metrics(y_test, y_pred, y_score)
    return {
        "label": label,
        "model": model,
        "y_pred": y_pred,
        "y_score": y_score,
        "metrics": metrics,
    }


def save_predictions(predictions: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)


def fit_meta_learner(
    jit_probs: Sequence[float],
    semantic_probs: Sequence[float],
    y_true: Sequence[int],
) -> LogisticRegression:
    meta_X = np.vstack([jit_probs, semantic_probs]).T
    meta_model = LogisticRegression(max_iter=2000, random_state=42)
    meta_model.fit(meta_X, y_true)
    return meta_model


def evaluate_meta_model(
    meta_model: LogisticRegression,
    jit_probs: Sequence[float],
    semantic_probs: Sequence[float],
    y_true: Sequence[int],
) -> Dict[str, float]:
    meta_X = np.vstack([jit_probs, semantic_probs]).T
    y_pred = meta_model.predict(meta_X)
    y_score = meta_model.predict_proba(meta_X)[:, 1]
    return compute_classification_metrics(y_true, y_pred, y_score)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    ensure_directories()

    if not INPUT_PATH.exists():
        logger.error("Frozen dataset not found at %s", INPUT_PATH)
        raise FileNotFoundError(f"Frozen dataset not found: {INPUT_PATH}")

    df = load_commits_csv(INPUT_PATH)
    df = expand_embedding_to_pca(df, PCA_FEATURES_384)

    missing = [col for col in ["commit_id", "project", "author_date", "buggy"] + JIT_FEATURES + PCA_FEATURES_384 if col not in df.columns]
    if missing:
        raise KeyError(f"Required columns missing from dataset: {missing}")

    df = df.sort_values("author_date").reset_index(drop=True)
    df = df.dropna(subset=JIT_FEATURES + PCA_FEATURES_384 + ["buggy"])

    X_train, X_test, y_train, y_test = temporal_train_test_split(
        df,
        timestamp_column="author_date",
        target_column="buggy",
        feature_columns=JIT_FEATURES + PCA_FEATURES_384,
        train_size=0.8,
    )
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    logger.info("Training JIT-only XGBoost")
    jit_result = train_xgb_experiment("jit_only", X_train, y_train, X_test, y_test, JIT_FEATURES)

    logger.info("Training semantic PCA-only XGBoost")
    pca_result = train_xgb_experiment("pca_only", X_train, y_train, X_test, y_test, PCA_FEATURES_25)

    logger.info("Training early fusion XGBoost")
    early_result = train_xgb_experiment("early_fusion", X_train, y_train, X_test, y_test, JIT_FEATURES + PCA_FEATURES_384)

    logger.info("Training top-25 fusion XGBoost")
    top25_result = train_xgb_experiment("top25_fusion", X_train, y_train, X_test, y_test, JIT_FEATURES + PCA_FEATURES_25)

    logger.info("Building stacking ensemble")
    internal_split_index = int(len(X_train) * 0.8)
    X_base_train = X_train.iloc[:internal_split_index].copy()
    X_meta_train = X_train.iloc[internal_split_index:].copy()
    y_base_train = y_train.iloc[:internal_split_index].copy()
    y_meta_train = y_train.iloc[internal_split_index:].copy()

    jit_base = train_xgb_experiment("jit_base", X_base_train, y_base_train, X_meta_train, y_meta_train, JIT_FEATURES)
    semantic_base = train_xgb_experiment(
        "semantic_base",
        X_base_train,
        y_base_train,
        X_meta_train,
        y_meta_train,
        PCA_FEATURES_25,
    )

    meta_learner = fit_meta_learner(jit_base["y_score"], semantic_base["y_score"], y_meta_train)
    save_json({"model_type": "logistic_regression", "trained_on": "stacking_meta_train"}, MODELS_DIR / "stacking_meta_model_info.json")

    jit_test_probs = jit_result["model"].predict_proba(X_test[JIT_FEATURES])[:, 1]
    semantic_test_probs = pca_result["model"].predict_proba(X_test[PCA_FEATURES_25])[:, 1]

    stacking_metrics = evaluate_meta_model(meta_learner, jit_test_probs, semantic_test_probs, y_test)
    stacking_test_probs = meta_learner.predict_proba(np.vstack([jit_test_probs, semantic_test_probs]).T)[:, 1]
    stacking_path = MODELS_DIR / "stacking_meta_model.joblib"
    from joblib import dump

    dump(meta_learner, stacking_path)

    results = {
        "JIT Only": jit_result["metrics"],
        "PCA Only": pca_result["metrics"],
        "Early Fusion": early_result["metrics"],
        "Top-25 Fusion": top25_result["metrics"],
        "Stacking Ensemble": stacking_metrics,
    }

    comparison_df = pd.DataFrame.from_dict(results, orient="index")
    comparison_df.index.name = "model"
    comparison_df.to_csv(METRICS_DIR / "metrics_comparison.csv")

    plot_roc_comparison(y_test, {
        "JIT Only": jit_result["y_score"],
        "PCA Only": pca_result["y_score"],
        "Early Fusion": early_result["y_score"],
        "Top-25 Fusion": top25_result["y_score"],
        "Stacking Ensemble": stacking_test_probs,
    }, PLOTS_DIR / "roc_auc_comparison.png")

    plot_pr_comparison(y_test, {
        "JIT Only": jit_result["y_score"],
        "PCA Only": pca_result["y_score"],
        "Early Fusion": early_result["y_score"],
        "Top-25 Fusion": top25_result["y_score"],
        "Stacking Ensemble": stacking_test_probs,
    }, PLOTS_DIR / "pr_auc_comparison.png")

    best_label = comparison_df["roc_auc"].idxmax()
    best_metrics = comparison_df.loc[best_label].to_dict()
    best_model = {
        "best_model": best_label,
        "metric": "roc_auc",
        "metrics": best_metrics,
    }
    save_json(best_model, METRICS_DIR / "best_model.json")

    predictions = pd.DataFrame({
        "commit_id": df.loc[X_test.index, "commit_id"].values,
        "author_date": df.loc[X_test.index, "author_date"].values,
        "buggy_true": y_test.values,
        "jit_score": np.asarray(jit_result["y_score"]),
        "pca_score": np.asarray(pca_result["y_score"]),
        "early_score": np.asarray(early_result["y_score"]),
        "top25_score": np.asarray(top25_result["y_score"]),
        "stacking_score": stacking_test_probs,
    })
    save_predictions(predictions, PREDICTIONS_DIR / "experiment_7_predictions.csv")

    logger.info("Experiment 7 complete. Results saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
