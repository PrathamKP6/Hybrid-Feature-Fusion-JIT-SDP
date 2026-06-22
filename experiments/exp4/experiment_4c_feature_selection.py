"""Experiment 4c: Optimal Top-K Semantic Feature Selection Study.

This script evaluates whether a smaller subset of the most important
PCA-reduced CodeBERT features can outperform the JIT-only baseline for
Just-in-Time software defect prediction.
"""
from __future__ import annotations

import ast
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, accuracy_score, f1_score, matthews_corrcoef, precision_score, precision_recall_curve, recall_score, roc_auc_score, roc_curve

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
RESULTS_DIR = PROJECT_ROOT / "results" / "exp4" / "experiment_4c"
BASELINE_JIT_ROC_AUC = 0.803
PREVIOUS_TOP25_ROC_AUC = 0.792

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
ALL_FEATURES = JIT_FEATURES + PCA_FEATURES
TOP_K_VALUES = [10, 15, 20, 25, 50, 100]


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    logger.info("Saved JSON to %s", path)


def save_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved CSV to %s", path)


def ensure_prob_array(y_score: np.ndarray) -> np.ndarray:
    arr = np.asarray(y_score)
    if arr.ndim == 2 and arr.shape[1] > 1:
        return arr[:, 1]
    return arr.ravel()


def parse_embedding_column(df: pd.DataFrame, pca_cols: List[str]) -> pd.DataFrame:
    missing = [c for c in pca_cols if c not in df.columns]
    if not missing:
        return df

    if "embedding" not in df.columns:
        raise KeyError(f"Missing PCA columns and no 'embedding' column available: {missing}")

    logger.info("Expanding embedding column into %d PCA features", len(pca_cols))

    def parse_value(value: object) -> List[float]:
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

    embedding_matrix = np.vstack(df["embedding"].apply(parse_value).values)
    if embedding_matrix.shape[1] != len(pca_cols):
        raise ValueError(f"Embedding dimension {embedding_matrix.shape[1]} does not match expected {len(pca_cols)}")

    emb_df = pd.DataFrame(embedding_matrix, columns=pca_cols, index=df.index)
    df = pd.concat([df.drop(columns=["embedding"]), emb_df], axis=1)
    logger.info("Expanded embedding values into PCA features")
    return df


def get_pca_ranking(train_X: pd.DataFrame, train_y: pd.Series) -> List[str]:
    logger.info("Training Random Forest on PCA features for ranking")
    model, _, _ = train_random_forest(
        train_X[PCA_FEATURES],
        train_y,
        train_X[PCA_FEATURES],
        train_y,
        model_path=RESULTS_DIR / "rf_pca_ranking.joblib",
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )
    importance_df = pd.DataFrame({"feature": PCA_FEATURES, "importance": model.feature_importances_})
    importance_df = importance_df.sort_values(by="importance", ascending=False)
    logger.info("Computed PCA feature importances for ranking")
    return importance_df["feature"].tolist()


def train_xgb_on_top_k(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    test_X: pd.DataFrame,
    test_y: pd.Series,
    selected_pca: List[str],
    k: int,
) -> Dict[str, object]:
    feature_subset = JIT_FEATURES + selected_pca
    label = f"xgb_top_{k}"

    model, y_pred, y_score = train_xgboost(
        train_X[feature_subset],
        train_y,
        test_X[feature_subset],
        test_y,
        model_path=RESULTS_DIR / f"{label}.joblib",
        n_estimators=100,
        learning_rate=0.1,
        scale_pos_weight=float((len(train_y) - train_y.sum()) / train_y.sum()) if train_y.sum() > 0 else 1.0,
        random_state=42,
    )

    y_score = ensure_prob_array(y_score)
    metrics = compute_classification_metrics(test_y, y_pred, y_score)
    metrics.update({"k": k, "model": "xgb", "selection": "rf_pca_importance"})
    return metrics


def plot_bar(metric_name: str, df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(df["k"].astype(str), df[metric_name], color="#4C72B0")
    plt.title(metric_name.replace("_", " ").upper())
    plt.xlabel("Top K PCA features")
    plt.ylabel(metric_name.replace("_", " ").upper())
    for i, value in enumerate(df[metric_name]):
        plt.text(i, value + 0.005, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    save_csv(output_path.with_suffix(".csv"), pd.DataFrame()) if False else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info("Saved plot to %s", output_path)


def main() -> None:
    ensure_results_dir()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Frozen sample not found: {INPUT_PATH}")

    df = load_commits_csv(INPUT_PATH)
    df = parse_embedding_column(df, PCA_FEATURES)

    required_columns = ["commit_id", "project", "author_date", "buggy"] + ALL_FEATURES
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    df = df[required_columns].copy()
    df = df.sort_values("author_date").reset_index(drop=True)
    df = df.dropna(subset=ALL_FEATURES + ["buggy"])

    X_train, X_test, y_train, y_test = temporal_train_test_split(
        df,
        timestamp_column="author_date",
        target_column="buggy",
        feature_columns=ALL_FEATURES,
        train_size=0.8,
    )
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    selected_pca_ranking = get_pca_ranking(X_train, y_train)
    results = []

    for k in TOP_K_VALUES:
        selected_pca = selected_pca_ranking[:k]
        metrics = train_xgb_on_top_k(X_train, y_train, X_test, y_test, selected_pca, k)
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="roc_auc", ascending=False).reset_index(drop=True)
    save_csv(RESULTS_DIR / "topk_experiment_results.csv", results_df)

    plot_bar("roc_auc", results_df, RESULTS_DIR / "roc_auc_comparison.png")
    plot_bar("pr_auc", results_df, RESULTS_DIR / "pr_auc_comparison.png")

    best = results_df.iloc[0].to_dict()
    best_config = {
        "best_k": int(best["k"]),
        "best_roc_auc": float(best["roc_auc"]),
        "best_pr_auc": float(best["pr_auc"]),
        "best_f1": float(best["f1"]),
        "best_mcc": float(best["mcc"]),
        "best_precision": float(best["precision"]),
        "best_recall": float(best["recall"]),
        "best_accuracy": float(best["accuracy"]),
        "difference_from_jit_baseline": float(best["roc_auc"] - BASELINE_JIT_ROC_AUC),
        "difference_from_previous_top25": float(best["roc_auc"] - PREVIOUS_TOP25_ROC_AUC),
    }
    save_json(RESULTS_DIR / "best_topk_config.json", best_config)

    print("Best K:", best_config["best_k"])
    print("Best ROC-AUC:", best_config["best_roc_auc"])
    print("Difference from JIT-only baseline (0.803):", best_config["difference_from_jit_baseline"])
    print("Difference from previous Top-25 result (0.792):", best_config["difference_from_previous_top25"])


if __name__ == "__main__":
    main()
