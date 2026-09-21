"""Focused Exp4 optimization pass using only defensible train-only variants."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.final_experiment_pipeline import (
    DATA_PATH,
    JIT_FEATURES,
    PCA_FEATURES,
    PROJECT_ROOT,
    create_or_load_split,
    metrics,
    prepare_features,
    select_threshold,
    validate_dataset,
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "final_experiment_4_optimization"
SEED = 42


def xgb_parameters(scale_pos_weight: float) -> dict[str, object]:
    return {
        "objective": "binary:logistic",
        "n_estimators": 100,
        "learning_rate": 0.1,
        "eval_metric": "logloss",
        "scale_pos_weight": scale_pos_weight,
        "random_state": SEED,
        "n_jobs": -1,
    }


def main() -> None:
    started = time.perf_counter()
    df = validate_dataset()
    splits = create_or_load_split(df)
    all_features = JIT_FEATURES + PCA_FEATURES
    prepared, medians = prepare_features(splits, all_features)
    train = splits["train"]
    y_train = train["buggy"]
    y_validation = splits["validation"]["buggy"]
    y_test = splits["test"]["buggy"]
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    variants: dict[str, list[str]] = {
        "A_full_early_fusion_396": all_features,
        "B_jit_plus_top50_pca_62": JIT_FEATURES + PCA_FEATURES[:50],
    }
    mi_scores = mutual_info_classif(prepared["train"][all_features], y_train, random_state=SEED)
    mi_ranked = [feature for _, feature in sorted(zip(mi_scores, all_features), key=lambda item: (-item[0], item[1]))]
    variants["C_train_mi_top25"] = mi_ranked[:25]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in ("metrics", "models", "predictions"):
        (OUTPUT_DIR / subdir).mkdir(exist_ok=True)

    validation_records: dict[str, dict[str, object]] = {}
    models: dict[str, XGBClassifier] = {}
    validation_probabilities: dict[str, object] = {}
    thresholds: dict[str, float] = {}
    for name, features in variants.items():
        model = XGBClassifier(**xgb_parameters(scale_pos_weight))
        model.fit(prepared["train"][features], y_train)
        validation_probability = model.predict_proba(prepared["validation"][features])[:, 1]
        threshold, validation_metrics = select_threshold(y_validation, validation_probability)
        models[name] = model
        validation_probabilities[name] = validation_probability
        thresholds[name] = threshold
        validation_records[name] = {
            "features": features,
            "feature_count": len(features),
            "threshold": threshold,
            "metrics": validation_metrics,
        }
        joblib.dump(model, OUTPUT_DIR / "models" / f"{name}.joblib")
        pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values("importance", ascending=False).to_csv(OUTPUT_DIR / "metrics" / f"{name}_feature_importance.csv", index=False)

    best_name = max(validation_records, key=lambda name: (validation_records[name]["metrics"]["f1"], validation_records[name]["metrics"]["mcc"], -len(validation_records[name]["features"])))
    best_features = variants[best_name]
    best_model = models[best_name]
    best_test_probability = best_model.predict_proba(prepared["test"][best_features])[:, 1]
    final_test_metrics = metrics(y_test, best_test_probability, thresholds[best_name])
    prediction_frame = splits["test"][["commit_id", "author_date", "project", "buggy"]].copy()
    prediction_frame.rename(columns={"buggy": "actual_buggy"}, inplace=True)
    prediction_frame["predicted_probability"] = best_test_probability
    prediction_frame["predicted_label"] = (best_test_probability >= thresholds[best_name]).astype(int)
    prediction_frame["model"] = "xgboost"
    prediction_frame["variant"] = best_name
    prediction_frame["threshold"] = thresholds[best_name]
    prediction_frame.to_csv(OUTPUT_DIR / "predictions" / "final_test_predictions.csv", index=False)

    summary = {
        "dataset_path": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "split_manifest": "results/final_shared_split",
        "methodology": "XGBoost-only focused Exp4 pass; train-only MI; validation F1 selection; one final test evaluation",
        "jit_features": JIT_FEATURES,
        "pca_features": len(PCA_FEATURES),
        "scale_pos_weight": scale_pos_weight,
        "xgboost_parameters": xgb_parameters(scale_pos_weight),
        "train_only_jit_medians": medians,
        "variants": validation_records,
        "selected_variant": best_name,
        "selected_features": best_features,
        "selected_validation_metrics": validation_records[best_name]["metrics"],
        "final_test_metrics": final_test_metrics,
        "runtime_seconds": time.perf_counter() - started,
    }
    (OUTPUT_DIR / "metrics" / "optimization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "metrics" / "validation_variant_metrics.json").write_text(json.dumps(validation_records, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "metrics" / "final_test_metrics.json").write_text(json.dumps({"xgboost": final_test_metrics}, indent=2), encoding="utf-8")
    print(json.dumps({"selected_variant": best_name, "validation": validation_records[best_name], "final_test": final_test_metrics, "runtime_seconds": summary["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()