"""Experiment 1: JIT features only pipeline.

This script loads the multilanguage commit dataset, cleans and subsamples it,
performs a temporal train/test split, trains Random Forest and XGBoost models,
and saves metrics, predictions, and visualizations.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from preprocessing.clean_data import clean_commits_df
from preprocessing.load_data import load_commits_csv
from preprocessing.sampling import create_stratified_subset
from preprocessing.temporal_split import temporal_train_test_split
from models.train_random_forest import train_random_forest
from models.train_xgboost import train_xgboost
from utils.metrics import compute_classification_metrics
from utils.paths import EXPERIMENT1_DIR, FINAL_DATASET_CSV


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = FINAL_DATASET_CSV
EXPERIMENT_DIR = EXPERIMENT1_DIR
METRICS_DIR = EXPERIMENT_DIR / "metrics"
PREDICTIONS_DIR = EXPERIMENT_DIR / "predictions"
PLOTS_DIR = EXPERIMENT_DIR / "plots"
MODELS_DIR = EXPERIMENT_DIR / "models"

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


def ensure_directories() -> None:
    for path in [METRICS_DIR, PREDICTIONS_DIR, PLOTS_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved JSON to %s", path)


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


def plot_roc_pr(
    y_true: pd.Series,
    scores: Dict[str, pd.Series],
    output_path: Path,
) -> None:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for label, y_score in scores.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for label, y_score in scores.items():
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        plt.plot(recall, precision, label=label)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info("Saved ROC/PR plot to %s", output_path)


def plot_feature_importance(
    model: object,
    feature_names: List[str],
    output_path: Path,
    title: str,
) -> None:
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model %s does not have feature_importances_", model.__class__.__name__)
        return

    importances = model.feature_importances_
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values(by="importance", ascending=False)

    plt.figure(figsize=(8, 6))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info("Saved feature importance plot to %s", output_path)


def main() -> None:
    ensure_directories()

    logger.info("Loading dataset from %s", DATA_PATH)
    df = load_commits_csv(DATA_PATH)

    logger.info("Cleaning dataset")
    df_clean, clean_report = clean_commits_df(df, jit_feature_columns=JIT_FEATURES, target_column="buggy")
    logger.info("Cleaning report: %s", clean_report)

    logger.info("Creating stratified subset for years 2018-2026")
    df_sampled = create_stratified_subset(df_clean)
    logger.info("Sampled dataset size: %d", len(df_sampled))

    logger.info("Performing temporal train/test split")
    X_train, X_test, y_train, y_test = temporal_train_test_split(
        df_sampled,
        timestamp_column="author_date",
        target_column="buggy",
        feature_columns=JIT_FEATURES,
        train_size=0.8,
    )

    # Ensure y values are numeric 0/1 for metric functions and XGBoost
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    scale_pos_weight = 1.0
    if y_train.sum() > 0:
        scale_pos_weight = float((len(y_train) - y_train.sum()) / y_train.sum())
    logger.info("Computed scale_pos_weight=%s from training label distribution", scale_pos_weight)

    rf_model_path = MODELS_DIR / "experiment_1_rf.joblib"
    xgb_model_path = MODELS_DIR / "experiment_1_xgb.joblib"

    rf_model, rf_pred, rf_score = train_random_forest(
        X_train,
        y_train,
        X_test,
        y_test,
        model_path=rf_model_path,
    )

    xgb_model, xgb_pred, xgb_score = train_xgboost(
        X_train,
        y_train,
        X_test,
        y_test,
        model_path=xgb_model_path,
        scale_pos_weight=scale_pos_weight,
    )

    logger.info("Evaluating Random Forest model")
    rf_metrics = compute_classification_metrics(y_test, rf_pred, rf_score)
    logger.info("Evaluating XGBoost model")
    xgb_metrics = compute_classification_metrics(y_test, xgb_pred, xgb_score)

    metrics_path = METRICS_DIR / "random_forest_metrics.json"
    save_json(rf_metrics, metrics_path)
    metrics_path = METRICS_DIR / "xgboost_metrics.json"
    save_json(xgb_metrics, metrics_path)

    split_report = {
        "n_rows": len(df_sampled),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_ratio": 0.8,
        "test_ratio": 0.2,
        "train_buggy_distribution": y_train.value_counts().sort_index().to_dict(),
        "test_buggy_distribution": y_test.value_counts().sort_index().to_dict(),
    }
    save_json(split_report, METRICS_DIR / "split_report.json")

    save_json(clean_report, METRICS_DIR / "cleaning_report.json")

    combined_summary = {
        "experiment": "experiment_1_jit_only",
        "dataset_rows": len(df_sampled),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": JIT_FEATURES,
        "scale_pos_weight": scale_pos_weight,
        "models": {
            "random_forest": rf_metrics,
            "xgboost": xgb_metrics,
        },
    }
    save_json(combined_summary, METRICS_DIR / "summary.json")

    predictions_meta = df_sampled.loc[X_test.index, ["commit_id", "project", "author_date"]].copy()
    predictions_df = predictions_meta.assign(
        buggy_true=y_test.values,
        rf_pred=rf_pred.values,
        rf_score=rf_score.values,
        xgb_pred=xgb_pred.values,
        xgb_score=xgb_score.values,
    )
    predictions_path = PREDICTIONS_DIR / "experiment_1_jit_only_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    logger.info("Saved predictions to %s", predictions_path)

    save_feature_importance_csv(
        rf_model,
        JIT_FEATURES,
        METRICS_DIR / "experiment_1_rf_feature_importance.csv",
    )
    save_feature_importance_csv(
        xgb_model,
        JIT_FEATURES,
        METRICS_DIR / "experiment_1_xgb_feature_importance.csv",
    )

    plot_roc_pr(
        y_test,
        {"Random Forest": rf_score, "XGBoost": xgb_score},
        PLOTS_DIR / "experiment_1_jit_only_roc_pr.png",
    )

    plot_feature_importance(
        rf_model,
        JIT_FEATURES,
        PLOTS_DIR / "experiment_1_jit_only_rf_feature_importance.png",
        title="Random Forest Feature Importance",
    )
    plot_feature_importance(
        xgb_model,
        JIT_FEATURES,
        PLOTS_DIR / "experiment_1_jit_only_xgb_feature_importance.png",
        title="XGBoost Feature Importance",
    )

    logger.info("Experiment 1 completed successfully")


if __name__ == "__main__":
    main()
