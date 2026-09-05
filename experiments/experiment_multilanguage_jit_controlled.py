"""Controlled JIT-only XGBoost experiment for the fixed multi-language subset."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, precision_recall_curve
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.clean_data import clean_commits_df
from preprocessing.load_data import load_commits_csv
from utils.metrics import compute_classification_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = PROJECT_ROOT / "results" / "data" / "frozen_sample_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "multilanguage_experiment_1_controlled_v1"
TABLES_DIR = OUTPUT_DIR / "tables"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
MODELS_DIR = OUTPUT_DIR / "models"

SEED = 42
TRAIN_ROWS = 6000
VALIDATION_ROWS = 2000
TEST_ROWS = 2000
DEFAULT_THRESHOLD = 0.50
JIT_FEATURES = [
    "la", "ld", "nf", "ns", "nd", "ent", "ndev", "age", "nuc",
    "aexp", "arexp", "asexp", "fix",
]
TARGET_COLUMN = "buggy"
TIMESTAMP_COLUMN = "author_date"


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def split_chronologically(df: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    ordered = df.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True)
    expected = TRAIN_ROWS + VALIDATION_ROWS + TEST_ROWS
    if len(ordered) != expected:
        raise ValueError(f"Expected exactly {expected} rows, found {len(ordered)}")
    train_end = TRAIN_ROWS
    validation_end = TRAIN_ROWS + VALIDATION_ROWS
    return (
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:validation_end].copy(),
        ordered.iloc[validation_end:].copy(),
    )


def build_model(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=SEED,
    )


def predict_model(
    model: XGBClassifier,
    frame: pd.DataFrame,
) -> np.ndarray:
    return model.predict_proba(frame[JIT_FEATURES])[:, 1]


def probability_summary(scores: Iterable[float]) -> Dict[str, float]:
    values = np.asarray(list(scores), dtype=float)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "p99": float(np.quantile(values, 0.99)),
    }


def expected_calibration_error(y_true: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (scores >= lower) & (scores < upper)
        if upper == 1.0:
            mask |= scores == upper
        if not np.any(mask):
            continue
        error += mask.sum() / total * abs(y_true[mask].mean() - scores[mask].mean())
    return float(error)


def evaluate_stage(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    description: str,
    scale_pos_weight: float,
) -> Dict[str, Any]:
    predictions = (scores >= threshold).astype(int)
    metrics = compute_classification_metrics(y_true, predictions, scores)
    metrics.update(
        {
            "description": description,
            "scale_pos_weight": float(scale_pos_weight),
            "threshold": float(threshold),
            "brier": float(brier_score_loss(y_true, scores)),
            "ece": expected_calibration_error(y_true, scores),
            "predicted_positive_count": int(predictions.sum()),
            "predicted_positive_rate": float(predictions.mean()),
            "predicted_class_distribution": {
                "0": int((predictions == 0).sum()),
                "1": int((predictions == 1).sum()),
            },
            "probability_distribution": probability_summary(scores),
        }
    )
    return metrics


def threshold_table(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.round(np.arange(0.01, 1.00, 0.01), 2):
        predictions = (scores >= threshold).astype(int)
        metrics = compute_classification_metrics(y_true, predictions, scores)
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "mcc": metrics["mcc"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "predicted_positive_rate": float(predictions.mean()),
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(table: pd.DataFrame) -> float:
    best = table.sort_values(
        by=["f1", "mcc", "threshold"],
        ascending=[False, False, True],
        kind="mergesort",
    ).iloc[0]
    return float(best["threshold"])


def main() -> None:
    for directory in (TABLES_DIR, PREDICTIONS_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    logger.info("Loading fixed multi-language subset from %s", DATA_PATH)
    df = load_commits_csv(DATA_PATH)
    df_clean, cleaning_report = clean_commits_df(
        df,
        jit_feature_columns=JIT_FEATURES,
        target_column=TARGET_COLUMN,
    )
    train, validation, test = split_chronologically(df_clean)

    y_train = train[TARGET_COLUMN].astype(int).to_numpy()
    y_validation = validation[TARGET_COLUMN].astype(int).to_numpy()
    y_test = test[TARGET_COLUMN].astype(int).to_numpy()
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    logger.info("Training-only scale_pos_weight=%s", scale_pos_weight)

    models = {
        "baseline": build_model(1.0),
        "class_weighted": build_model(scale_pos_weight),
    }
    scores: Dict[str, Dict[str, np.ndarray]] = {}
    for name, model in models.items():
        model.fit(train[JIT_FEATURES], y_train)
        scores[name] = {
            "validation": predict_model(model, validation),
            "test": predict_model(model, test),
        }
        joblib.dump(model, MODELS_DIR / f"{name}_xgboost.joblib")

    weighted_validation_table = threshold_table(
        y_validation,
        scores["class_weighted"]["validation"],
    )
    weighted_validation_table.to_csv(TABLES_DIR / "validation_threshold_table_raw.csv", index=False)
    selected_threshold = choose_threshold(weighted_validation_table)

    stage_results = {
        "baseline": evaluate_stage(
            y_test,
            scores["baseline"]["test"],
            DEFAULT_THRESHOLD,
            "Unweighted XGBoost, raw probability at threshold 0.50",
            1.0,
        ),
        "class_weighted": evaluate_stage(
            y_test,
            scores["class_weighted"]["test"],
            DEFAULT_THRESHOLD,
            "Training-weighted XGBoost, raw probability at threshold 0.50",
            scale_pos_weight,
        ),
        "class_weighted_threshold_tuned": evaluate_stage(
            y_test,
            scores["class_weighted"]["test"],
            selected_threshold,
            "Training-weighted XGBoost, validation-tuned raw threshold",
            scale_pos_weight,
        ),
    }

    comparison = pd.DataFrame(
        [
            {
                "stage": name,
                **{key: value for key, value in result.items() if not isinstance(value, dict)},
            }
            for name, result in stage_results.items()
        ]
    )
    comparison.to_csv(TABLES_DIR / "final_test_comparison.csv", index=False)

    validation_summary = evaluate_stage(
        y_validation,
        scores["class_weighted"]["validation"],
        selected_threshold,
        "Training-weighted XGBoost, validation-selected threshold",
        scale_pos_weight,
    )
    save_json(validation_summary, TABLES_DIR / "validation_threshold_selection.json")

    prediction_frame = test[["commit_id", "project", TIMESTAMP_COLUMN, TARGET_COLUMN]].copy()
    prediction_frame.rename(columns={TARGET_COLUMN: "buggy_true"}, inplace=True)
    for name, stage_scores in scores.items():
        prediction_frame[f"{name}_score"] = stage_scores["test"]
        prediction_frame[f"{name}_pred"] = (
            stage_scores["test"] >= DEFAULT_THRESHOLD
        ).astype(int)
    prediction_frame["class_weighted_threshold_tuned_pred"] = (
        scores["class_weighted"]["test"] >= selected_threshold
    ).astype(int)
    prediction_frame.to_csv(PREDICTIONS_DIR / "final_test_predictions.csv", index=False)

    split_report = {
        "dataset_path": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "rows": len(df_clean),
        "features": JIT_FEATURES,
        "target": TARGET_COLUMN,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "train_target_distribution": pd.Series(y_train).value_counts().sort_index().to_dict(),
        "validation_target_distribution": pd.Series(y_validation).value_counts().sort_index().to_dict(),
        "test_target_distribution": pd.Series(y_test).value_counts().sort_index().to_dict(),
        "train_date_range": [str(train[TIMESTAMP_COLUMN].min()), str(train[TIMESTAMP_COLUMN].max())],
        "validation_date_range": [str(validation[TIMESTAMP_COLUMN].min()), str(validation[TIMESTAMP_COLUMN].max())],
        "test_date_range": [str(test[TIMESTAMP_COLUMN].min()), str(test[TIMESTAMP_COLUMN].max())],
    }
    save_json(split_report, TABLES_DIR / "split_report.json")
    save_json(cleaning_report, TABLES_DIR / "cleaning_report.json")
    save_json(
        {
            "experiment": "multilanguage_experiment_1_controlled_v1",
            "seed": SEED,
            "default_threshold": DEFAULT_THRESHOLD,
            "selected_threshold": selected_threshold,
            "scale_pos_weight": scale_pos_weight,
            "features": JIT_FEATURES,
            "target": TARGET_COLUMN,
            "models": stage_results,
        },
        OUTPUT_DIR / "summary.json",
    )
    logger.info("Completed multi-language controlled experiment: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
