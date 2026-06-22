"""Experiment 4c: Feature selection for JIT + CodeBERT fusion.

This script evaluates whether feature selection improves the performance of a
JIT + PCA-reduced CodeBERT fusion model. It compares three selection methods:
Random Forest importance, XGBoost importance, and mutual information. It
builds feature subsets of top 25, 50, 100, 200, and all features, then trains
Random Forest and XGBoost on each subset.
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
from sklearn.feature_selection import mutual_info_classif
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
OUTPUT_DIR = PROJECT_ROOT / "results" / "experiment_4c"
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
TOP_K_VALUES = [25, 50, 100, 200, len(FEATURE_COLUMNS)]
SELECTION_METHODS = ["rf_importance", "xgb_importance", "mutual_info"]


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


def plot_metric_line_comparison(results: Dict[str, Dict[str, Dict[str, float]]], output_path: Path) -> None:
    metrics = ["roc_auc", "pr_auc", "f1", "mcc"]
    plt.figure(figsize=(16, 12))

    for idx, metric in enumerate(metrics, start=1):
        plt.subplot(2, 2, idx)
        for method in SELECTION_METHODS:
            for model_type in ["rf", "xgb"]:
                label = f"{model_type.upper()}_{method}"
                values = [results[method][k][model_type][metric] for k in TOP_K_VALUES]
                plt.plot(TOP_K_VALUES, values, marker="o", label=label)
        plt.title(metric.replace("_", " ").upper())
        plt.xlabel("Top k features")
        plt.ylabel(metric.replace("_", " ").upper())
        plt.xticks(TOP_K_VALUES, [str(k) if k != len(FEATURE_COLUMNS) else "all" for k in TOP_K_VALUES])
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(fontsize="small")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info("Saved metric line comparison plot to %s", output_path)


def plot_best_roc_by_k(results: Dict[str, Dict[str, Dict[str, float]]], output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for method in SELECTION_METHODS:
        values = [max(results[method][k]["rf"]["roc_auc"], results[method][k]["xgb"]["roc_auc"]) for k in TOP_K_VALUES]
        plt.plot(TOP_K_VALUES, values, marker="o", label=method)
    plt.title("Best ROC-AUC by Top k Features")
    plt.xlabel("Top k features")
    plt.ylabel("ROC-AUC")
    plt.xticks(TOP_K_VALUES, [str(k) if k != len(FEATURE_COLUMNS) else "all" for k in TOP_K_VALUES])
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info("Saved best ROC-AUC by k plot to %s", output_path)


def get_feature_ranking_by_importance(model: object, feature_names: List[str]) -> List[str]:
    importance_df = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
    importance_df = importance_df.sort_values(by="importance", ascending=False)
    return importance_df["feature"].tolist()


def get_feature_ranking_by_mi(X: pd.DataFrame, y: pd.Series, feature_names: List[str]) -> List[str]:
    mi_values = mutual_info_classif(X[feature_names], y, discrete_features=False, random_state=42)
    importance_df = pd.DataFrame({"feature": feature_names, "importance": mi_values})
    importance_df = importance_df.sort_values(by="importance", ascending=False)
    return importance_df["feature"].tolist()


def train_and_evaluate_subset(
    df: pd.DataFrame,
    feature_subset: List[str],
    model_type: str,
    label: str,
) -> Dict[str, object]:
    X_train, X_test, y_train, y_test = temporal_train_test_split(
        df,
        timestamp_column="author_date",
        target_column="buggy",
        feature_columns=feature_subset,
        train_size=0.8,
    )
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    scale_pos_weight = 1.0
    if y_train.sum() > 0:
        scale_pos_weight = float((len(y_train) - y_train.sum()) / y_train.sum())

    model_path = MODELS_DIR / f"{model_type}_{label}.joblib"
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
    save_json(metrics, METRICS_DIR / f"{model_type}_{label}_metrics.json")

    predictions_df = pd.DataFrame(
        {
            "commit_id": df.loc[X_test.index, "commit_id"].values,
            "project": df.loc[X_test.index, "project"].values,
            "author_date": df.loc[X_test.index, "author_date"].values,
            "buggy_true": y_test.values,
            "y_pred": y_pred.values,
            "y_score": y_score,
        }
    )
    predictions_df.to_csv(PREDICTIONS_DIR / f"{model_type}_{label}_predictions.csv", index=False)

    return {
        "metrics": metrics,
        "model": model,
        "feature_subset": feature_subset,
    }


def summarize_best_configuration(results: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, object]:
    best = None
    best_roc = -1.0
    for method in SELECTION_METHODS:
        for k in TOP_K_VALUES:
            for model_type in ["rf", "xgb"]:
                roc = results[method][k][model_type]["roc_auc"]
                if roc > best_roc:
                    best_roc = roc
                    best = {
                        "method": method,
                        "k": k,
                        "model": model_type,
                        "roc_auc": roc,
                        "pr_auc": results[method][k][model_type]["pr_auc"],
                        "f1": results[method][k][model_type]["f1"],
                        "mcc": results[method][k][model_type]["mcc"],
                    }
    if best is None:
        raise RuntimeError("No best configuration found")
    complexity = "all" if best["k"] == len(FEATURE_COLUMNS) else str(best["k"])
    return {
        "best_method": best["method"],
        "best_model": best["model"],
        "best_k": complexity,
        "best_roc_auc": best["roc_auc"],
        "best_pr_auc": best["pr_auc"],
        "best_f1": best["f1"],
        "best_mcc": best["mcc"],
    }


def main() -> None:
    logger.info("Experiment 4c (feature selection) starting")
    ensure_directories()

    if not INPUT_PATH.exists():
        logger.error("Frozen sample not found at %s", INPUT_PATH)
        raise FileNotFoundError(f"Frozen sample not found: {INPUT_PATH}")

    df = load_commits_csv(INPUT_PATH)
    logger.info("Loaded frozen sample: %d rows, %d columns", len(df), len(df.columns))

    df = expand_embedding_to_pca(df, PCA_FEATURES)

    missing_cols = [col for col in ["commit_id", "project", "author_date", "buggy"] + FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        logger.error("Missing required columns: %s", missing_cols[:20])
        raise KeyError(f"Missing required columns: {missing_cols}")

    df_reduced = df[["commit_id", "project", "author_date", "buggy"] + FEATURE_COLUMNS].copy()
    df_reduced = df_reduced.sort_values("author_date").reset_index(drop=True)
    n_missing = int(df_reduced[FEATURE_COLUMNS].isna().any(axis=1).sum())
    if n_missing > 0:
        logger.warning("Dropping %d rows with missing feature values", n_missing)
        df_reduced = df_reduced.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    X_train_full, X_test_full, y_train_full, y_test_full = temporal_train_test_split(
        df_reduced,
        timestamp_column="author_date",
        target_column="buggy",
        feature_columns=FEATURE_COLUMNS,
        train_size=0.8,
    )
    y_train_full = y_train_full.astype(int)
    y_test_full = y_test_full.astype(int)

    # Train full models only for feature ranking and baseline comparison.
    logger.info("Training full Random Forest for feature ranking")
    rf_full_model, _, _ = train_random_forest(
        X_train_full,
        y_train_full,
        X_test_full,
        y_test_full,
        model_path=MODELS_DIR / "rf_full_for_selection.joblib",
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )
    logger.info("Training full XGBoost for feature ranking")
    xgb_full_model, _, _ = train_xgboost(
        X_train_full,
        y_train_full,
        X_test_full,
        y_test_full,
        model_path=MODELS_DIR / "xgb_full_for_selection.joblib",
        n_estimators=100,
        learning_rate=0.1,
        scale_pos_weight=float((len(y_train_full) - y_train_full.sum()) / y_train_full.sum()) if y_train_full.sum() > 0 else 1.0,
        random_state=42,
    )

    rankings: Dict[str, List[str]] = {
        "rf_importance": get_feature_ranking_by_importance(rf_full_model, FEATURE_COLUMNS),
        "xgb_importance": get_feature_ranking_by_importance(xgb_full_model, FEATURE_COLUMNS),
        "mutual_info": get_feature_ranking_by_mi(X_train_full, y_train_full, FEATURE_COLUMNS),
    }

    results: Dict[str, Dict[int, Dict[str, Dict[str, float]]]] = {
        method: {} for method in SELECTION_METHODS
    }

    for method in SELECTION_METHODS:
        logger.info("Computing results for selection method %s", method)
        for k in TOP_K_VALUES:
            label_k = "all" if k == len(FEATURE_COLUMNS) else str(k)
            top_features = rankings[method] if k == len(FEATURE_COLUMNS) else rankings[method][:k]
            results[method][k] = {"rf": {}, "xgb": {}}

            rf_result = train_and_evaluate_subset(df_reduced, top_features, "rf", f"rf_{method}_{label_k}")
            xgb_result = train_and_evaluate_subset(df_reduced, top_features, "xgb", f"xgb_{method}_{label_k}")

            results[method][k]["rf"] = rf_result["metrics"]
            results[method][k]["xgb"] = xgb_result["metrics"]

            save_json(
                {
                    "method": method,
                    "k": label_k,
                    "rf_metrics": rf_result["metrics"],
                    "xgb_metrics": xgb_result["metrics"],
                },
                METRICS_DIR / f"feature_selection_{method}_{label_k}_metrics.json",
            )

    save_json({method: rankings[method] for method in SELECTION_METHODS}, METRICS_DIR / "feature_rankings.json")

    plot_metric_line_comparison(results, PLOTS_DIR / "experiment_4c_metric_line_comparison.png")
    plot_best_roc_by_k(results, PLOTS_DIR / "experiment_4c_best_roc_by_k.png")

    summary = summarize_best_configuration(results)
    save_json(summary, METRICS_DIR / "feature_selection_summary.json")

    logger.info("Experiment 4c completed successfully")


if __name__ == "__main__":
    main()
