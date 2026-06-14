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


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "final_multilanguage_dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"

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
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved JSON to %s", path)


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

    metrics_path = METRICS_DIR / "experiment_1_jit_only_metrics.json"
    save_json({"random_forest": rf_metrics, "xgboost": xgb_metrics}, metrics_path)

    predictions_df = X_test.copy()
    predictions_df["buggy_true"] = y_test.values
    predictions_df["rf_pred"] = rf_pred.values
    predictions_df["rf_score"] = rf_score.values
    predictions_df["xgb_pred"] = xgb_pred.values
    predictions_df["xgb_score"] = xgb_score.values
    predictions_path = PREDICTIONS_DIR / "experiment_1_jit_only_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    logger.info("Saved predictions to %s", predictions_path)

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
