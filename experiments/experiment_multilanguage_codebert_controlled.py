"""Controlled CodeBERT-only XGBoost experiment for the fixed multi-language subset."""
from __future__ import annotations

import ast
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, accuracy_score, brier_score_loss, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.clean_data import clean_commits_df
from preprocessing.load_data import load_commits_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = PROJECT_ROOT / "results" / "data" / "frozen_sample_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "multilanguage_experiment_2_controlled_v1"
TABLES_DIR = OUTPUT_DIR / "tables"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
MODELS_DIR = OUTPUT_DIR / "models"
SEED = 42
FEATURES = [f"pca_{index}" for index in range(1, 385)]
JIT_FEATURES = ["la", "ld", "nf", "ns", "nd", "ent", "ndev", "age", "nuc", "aexp", "arexp", "asexp", "fix"]


def save_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else value), encoding="utf-8")


def metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float, description: str, scale_pos_weight: float) -> Dict[str, Any]:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(threshold),
        "scale_pos_weight": float(scale_pos_weight),
        "description": description,
        "predicted_positive_count": int(predictions.sum()),
        "predicted_positive_rate": float(predictions.mean()),
        "brier": float(brier_score_loss(y_true, scores)),
    }


def threshold_table(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.round(np.arange(0.01, 1.00, 0.01), 2):
        result = metrics(y_true, scores, float(threshold), "validation", 1.0)
        rows.append(result)
    return pd.DataFrame(rows)


def main() -> None:
    for directory in (TABLES_DIR, PREDICTIONS_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    df = load_commits_csv(DATA_PATH)
    embedding_values = df["embedding"].map(ast.literal_eval)
    embedding_matrix = np.vstack(embedding_values.to_numpy())
    if embedding_matrix.shape != (len(df), len(FEATURES)):
        raise ValueError(f"Expected ({len(df)}, {len(FEATURES)}) embeddings, found {embedding_matrix.shape}")
    pca_frame = pd.DataFrame(embedding_matrix, columns=FEATURES, index=df.index)
    df = pd.concat([df.drop(columns=["embedding"]), pca_frame], axis=1)
    df, cleaning_report = clean_commits_df(df, jit_feature_columns=JIT_FEATURES, target_column="buggy")
    ordered = df.sort_values("author_date", kind="mergesort").reset_index(drop=True)
    train, validation, test = ordered.iloc[:6000], ordered.iloc[6000:8000], ordered.iloc[8000:10000]
    y_train = train["buggy"].astype(int).to_numpy()
    y_validation = validation["buggy"].astype(int).to_numpy()
    y_test = test["buggy"].astype(int).to_numpy()
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    models = {
        "baseline": XGBClassifier(n_estimators=100, learning_rate=0.1, scale_pos_weight=1.0, use_label_encoder=False, eval_metric="logloss", random_state=SEED),
        "class_weighted": XGBClassifier(n_estimators=100, learning_rate=0.1, scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric="logloss", random_state=SEED),
    }
    scores = {}
    for name, model in models.items():
        model.fit(train[FEATURES], y_train)
        scores[name] = {"validation": model.predict_proba(validation[FEATURES])[:, 1], "test": model.predict_proba(test[FEATURES])[:, 1]}
        joblib.dump(model, MODELS_DIR / f"{name}_xgboost.joblib")

    validation_table = threshold_table(y_validation, scores["class_weighted"]["validation"])
    validation_table.to_csv(TABLES_DIR / "validation_threshold_table_raw.csv", index=False)
    best_row = validation_table.sort_values(["f1", "mcc", "threshold"], ascending=[False, False, True], kind="mergesort").iloc[0]
    selected_threshold = float(best_row["threshold"])

    results = {
        "baseline": metrics(y_test, scores["baseline"]["test"], 0.5, "Unweighted XGBoost at threshold 0.50", 1.0),
        "class_weighted": metrics(y_test, scores["class_weighted"]["test"], 0.5, "Training-weighted XGBoost at threshold 0.50", scale_pos_weight),
        "class_weighted_threshold_tuned": metrics(y_test, scores["class_weighted"]["test"], selected_threshold, "Training-weighted XGBoost at validation-tuned threshold", scale_pos_weight),
    }
    pd.DataFrame([{"stage": name, **result} for name, result in results.items()]).to_csv(TABLES_DIR / "final_test_comparison.csv", index=False)

    predictions = test[["commit_id", "project", "author_date", "buggy"]].rename(columns={"buggy": "buggy_true"}).copy()
    for name, stage_scores in scores.items():
        predictions[f"{name}_score"] = stage_scores["test"]
        predictions[f"{name}_pred"] = (stage_scores["test"] >= 0.5).astype(int)
    predictions["class_weighted_threshold_tuned_pred"] = (scores["class_weighted"]["test"] >= selected_threshold).astype(int)
    predictions.to_csv(PREDICTIONS_DIR / "final_test_predictions.csv", index=False)

    split_report = {
        "dataset_path": str(DATA_PATH.relative_to(PROJECT_ROOT)), "rows": len(df), "features": FEATURES,
        "target": "buggy", "train_rows": 6000, "validation_rows": 2000, "test_rows": 2000,
        "train_target_distribution": pd.Series(y_train).value_counts().sort_index().to_dict(),
        "validation_target_distribution": pd.Series(y_validation).value_counts().sort_index().to_dict(),
        "test_target_distribution": pd.Series(y_test).value_counts().sort_index().to_dict(),
    }
    save_json(split_report, TABLES_DIR / "split_report.json")
    save_json(cleaning_report, TABLES_DIR / "cleaning_report.json")
    save_json({"experiment": "multilanguage_experiment_2_controlled_v1", "scale_pos_weight": scale_pos_weight, "selected_threshold": selected_threshold, "results": results}, OUTPUT_DIR / "summary.json")
    logger.info("Completed Exp2: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
