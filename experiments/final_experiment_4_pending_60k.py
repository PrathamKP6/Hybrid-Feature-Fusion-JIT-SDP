"""Run the four pending full-dataset Exp4 fusion variants."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.final_experiment_pipeline import (  # noqa: E402
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

OUTPUT_DIR = PROJECT_ROOT / "results" / "final_experiment_4_pending_60k"
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


def save_variant(
    name: str,
    features: list[str],
    model_name: str,
    validation_metrics: dict[str, object],
    test_metrics: dict[str, object],
    validation_extra: dict[str, object],
    model: object,
    test_frame: pd.DataFrame,
    test_probability: np.ndarray,
    threshold: float,
) -> None:
    variant_dir = OUTPUT_DIR / name
    for subdir in ("metrics", "models", "predictions"):
        (variant_dir / subdir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, variant_dir / "models" / f"{model_name}.joblib")
    if hasattr(model, "feature_importances_"):
        pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values("importance", ascending=False).to_csv(variant_dir / "metrics" / "feature_importance.csv", index=False)
    prediction_frame = test_frame[["commit_id", "author_date", "project", "buggy"]].copy()
    prediction_frame.rename(columns={"buggy": "actual_buggy"}, inplace=True)
    prediction_frame["predicted_probability"] = test_probability
    prediction_frame["predicted_label"] = (test_probability >= threshold).astype(int)
    prediction_frame["model"] = model_name
    prediction_frame["threshold"] = threshold
    prediction_frame.to_csv(variant_dir / "predictions" / "final_test_predictions.csv", index=False)
    (variant_dir / "metrics" / "validation_metrics.json").write_text(json.dumps({**validation_metrics, **validation_extra}, indent=2), encoding="utf-8")
    (variant_dir / "metrics" / "final_test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    (variant_dir / "metrics" / "configuration.json").write_text(json.dumps({"variant": name, "model": model_name, "features": features, "feature_count": len(features), "dataset_path": str(DATA_PATH.relative_to(PROJECT_ROOT)), "split_manifest": "results/final_shared_split", "threshold": threshold}, indent=2), encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    df = validate_dataset()
    splits = create_or_load_split(df)
    all_features = JIT_FEATURES + PCA_FEATURES
    prepared, medians = prepare_features(splits, all_features)
    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]
    y_train = train["buggy"]
    y_validation = validation["buggy"]
    y_test = test["buggy"]
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    params = xgb_parameters(scale_pos_weight)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, object]] = {}
    models: dict[str, XGBClassifier] = {}

    def train_xgb(name: str, features: list[str]) -> tuple[XGBClassifier, np.ndarray, np.ndarray]:
        model = XGBClassifier(**params)
        model.fit(prepared["train"][features], y_train)
        val_probability = model.predict_proba(prepared["validation"][features])[:, 1]
        test_probability = model.predict_proba(prepared["test"][features])[:, 1]
        models[name] = model
        return model, val_probability, test_probability

    # 4B1: fixed first 25 PCA components plus all JIT features.
    b1_features = JIT_FEATURES + PCA_FEATURES[:25]
    b1_model, b1_val, b1_test = train_xgb("4B1_top25", b1_features)
    b1_threshold, b1_validation = select_threshold(y_validation, b1_val)
    b1_test_metrics = metrics(y_test, b1_test, b1_threshold)
    save_variant("4B1_top25", b1_features, "xgboost", b1_validation, b1_test_metrics, {"threshold_selection": "validation F1"}, b1_model, test, b1_test, b1_threshold)
    results["4B1_top25"] = {"validation": b1_validation, "final_test": b1_test_metrics, "threshold": b1_threshold, "feature_count": len(b1_features)}

    # 4C: mutual information is fitted only on training rows and labels.
    mi_scores = mutual_info_classif(prepared["train"][all_features], y_train, random_state=SEED)
    mi_ranked = [feature for _, feature in sorted(zip(mi_scores, all_features), key=lambda item: (-item[0], item[1]))]
    c_features = mi_ranked[:25]
    c_model, c_val, c_test = train_xgb("4C_mi_top25", c_features)
    c_threshold, c_validation = select_threshold(y_validation, c_val)
    c_test_metrics = metrics(y_test, c_test, c_threshold)
    c_extra = {"selection": "mutual_info_classif on training data only", "selected_features": c_features, "selected_jit_count": sum(feature in JIT_FEATURES for feature in c_features), "selected_pca_count": sum(feature in PCA_FEATURES for feature in c_features), "threshold_selection": "validation F1"}
    save_variant("4C_mi_top25", c_features, "xgboost", c_validation, c_test_metrics, c_extra, c_model, test, c_test, c_threshold)
    results["4C_mi_top25"] = {"validation": c_validation, "final_test": c_test_metrics, "threshold": c_threshold, **c_extra}

    # Shared base models for 4D and 4E.
    jit_model, jit_val, jit_test = train_xgb("4D_jit_base", JIT_FEATURES)
    codebert_model, cb_val, cb_test = train_xgb("4D_codebert_base", PCA_FEATURES)

    # 4D: validation predictions train the simple logistic meta-model.
    meta = LogisticRegression(solver="lbfgs", random_state=SEED)
    meta.fit(np.column_stack([jit_val, cb_val]), y_validation)
    stack_val = meta.predict_proba(np.column_stack([jit_val, cb_val]))[:, 1]
    stack_test = meta.predict_proba(np.column_stack([jit_test, cb_test]))[:, 1]
    d_threshold, d_validation = select_threshold(y_validation, stack_val)
    d_test_metrics = metrics(y_test, stack_test, d_threshold)
    d_extra = {"base_models": ["XGBoost JIT-only", "XGBoost CodeBERT-only"], "meta_model": "LogisticRegression(solver=lbfgs, random_state=42)", "threshold_selection": "validation F1", "validation_meta_fit": True}
    save_variant("4D_stacking", ["jit_base_probability", "codebert_base_probability"], "logistic_stacking_meta_model", d_validation, d_test_metrics, d_extra, meta, test, stack_test, d_threshold)
    results["4D_stacking"] = {"validation": d_validation, "final_test": d_test_metrics, "threshold": d_threshold, **d_extra}

    # 4E: select the JIT probability weight on a fixed 0.0..1.0 validation grid.
    blend_candidates: list[tuple[float, float, float, dict[str, object]]] = []
    for jit_weight in np.round(np.arange(0.0, 1.01, 0.1), 1):
        blend_val = jit_weight * jit_val + (1.0 - jit_weight) * cb_val
        blend_threshold, blend_metrics = select_threshold(y_validation, blend_val)
        blend_candidates.append((float(blend_metrics["f1"]), float(blend_metrics["mcc"]), float(jit_weight), {"threshold": blend_threshold, "metrics": blend_metrics}))
    _, _, e_jit_weight, e_choice = max(blend_candidates, key=lambda item: (item[0], item[1], -item[2]))
    e_cb_weight = 1.0 - e_jit_weight
    e_threshold = float(e_choice["threshold"])
    blend_test = e_jit_weight * jit_test + e_cb_weight * cb_test
    e_test_metrics = metrics(y_test, blend_test, e_threshold)
    e_extra = {"jit_weight": e_jit_weight, "codebert_weight": e_cb_weight, "weight_grid": [round(float(value), 1) for value in np.arange(0.0, 1.01, 0.1)], "threshold_selection": "validation F1"}
    save_variant("4E_probability_blend", ["jit_base_probability", "codebert_base_probability"], "probability_blend", e_choice["metrics"], e_test_metrics, e_extra, {"jit_model": jit_model, "codebert_model": codebert_model}, test, blend_test, e_threshold)
    results["4E_probability_blend"] = {"validation": e_choice["metrics"], "final_test": e_test_metrics, "threshold": e_threshold, **e_extra}

    summary = {"dataset_path": str(DATA_PATH.relative_to(PROJECT_ROOT)), "split_manifest": "results/final_shared_split", "scale_pos_weight": scale_pos_weight, "xgboost_parameters": params, "train_only_jit_medians": medians, "variants": results, "runtime_seconds": time.perf_counter() - started}
    (OUTPUT_DIR / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"runtime_seconds": summary["runtime_seconds"], "variants": results}, indent=2))


if __name__ == "__main__":
    from sklearn.feature_selection import mutual_info_classif

    main()