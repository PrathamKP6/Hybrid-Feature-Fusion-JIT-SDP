"""Controlled Experiment 4: multi-language JIT + CodeBERT fusion benchmark."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "results" / "data" / "frozen_sample_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "multilanguage_experiment_4_controlled_v1"
JIT_FEATURES: List[str] = [
    "la", "ld", "nf", "ns", "nd", "ent", "ndev", "age", "nuc", "aexp", "arexp", "asexp",
]
PCA_COLS = [f"pca_{i}" for i in range(1, 385)]
SEED = 42


def ensure_dirs() -> None:
    for sub in ("metrics", "predictions", "tables", "plots", "models"):
        (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)


def parse_author_date(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    numeric = pd.to_numeric(series, errors="coerce")
    if int(numeric.notna().sum()) >= max(1, int(len(series) * 0.1)):
        maximum = int(numeric.abs().max(skipna=True))
        unit = "ns" if maximum > 10**15 else "ms" if maximum > 10**12 else "s"
        return pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)
    return pd.to_datetime(series, errors="coerce", utc=True)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(round(threshold, 4)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "predicted_positive_rate": float(np.mean(y_pred)),
        "predicted_positive_count": int(y_pred.sum()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, candidates: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    best_threshold = 0.50
    best_metrics: Dict[str, Any] | None = None
    for threshold in candidates:
        current = compute_metrics(y_true, y_prob, float(threshold))
        if best_metrics is None or current["f1"] > best_metrics["f1"]:
            best_threshold, best_metrics = float(threshold), current
    return best_threshold, best_metrics or compute_metrics(y_true, y_prob, 0.50)


def train_xgb(x_train: pd.DataFrame, y_train: pd.Series, spw: float) -> XGBClassifier:
    model = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", learning_rate=0.1,
        n_estimators=100, scale_pos_weight=spw, random_state=SEED, n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return model


def main() -> None:
    ensure_dirs()
    logger.info("Loading fixed multi-language dataset from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df["author_date"] = parse_author_date(df["author_date"])
    df = df.sort_values("author_date", kind="mergesort").reset_index(drop=True)

    if "embedding" in df.columns and not all(column in df.columns for column in PCA_COLS):
        logger.info("Expanding embedding column into %d PCA columns", len(PCA_COLS))
        embeddings = np.vstack([json.loads(value) for value in df["embedding"]])
        if embeddings.shape != (len(df), len(PCA_COLS)):
            raise ValueError(f"Expected {(len(df), len(PCA_COLS))} embeddings, found {embeddings.shape}")
        df = pd.concat([df.drop(columns=["embedding"]), pd.DataFrame(embeddings, columns=PCA_COLS)], axis=1)

    all_features = JIT_FEATURES + PCA_COLS
    df = df[["commit_id", "project", "author_date", "buggy"] + all_features].dropna(subset=["author_date"]).copy()
    df["buggy"] = df["buggy"].astype(int)
    if len(df) != 10000:
        raise ValueError(f"Expected 10000 fixed-subset rows, found {len(df)}")

    train_df = df.iloc[:6000].copy()
    val_df = df.iloc[6000:8000].copy()
    test_df = df.iloc[8000:10000].copy()
    y_train = train_df["buggy"]
    y_val = val_df["buggy"].to_numpy()
    y_test = test_df["buggy"].to_numpy()
    train_neg = int((y_train == 0).sum())
    train_pos = int((y_train == 1).sum())
    spw = float(train_neg / train_pos)
    logger.info("Splits: train=%d, validation=%d, test=%d, scale_pos_weight=%.8f", len(train_df), len(val_df), len(test_df), spw)

    model_jit = train_xgb(train_df[JIT_FEATURES], y_train, spw)
    model_codebert = train_xgb(train_df[PCA_COLS], y_train, spw)
    early_features = JIT_FEATURES + PCA_COLS
    model_4a = train_xgb(train_df[early_features], y_train, spw)
    features_4b1 = JIT_FEATURES + PCA_COLS[:25]
    model_4b1 = train_xgb(train_df[features_4b1], y_train, spw)
    features_4b2 = JIT_FEATURES + PCA_COLS[:50]
    model_4b2 = train_xgb(train_df[features_4b2], y_train, spw)

    mi_scores = mutual_info_classif(train_df[all_features], y_train, random_state=SEED)
    mi_ranked = [feature for _, feature in sorted(zip(mi_scores, all_features), reverse=True)]
    top25_mi = mi_ranked[:25]
    model_4c = train_xgb(train_df[top25_mi], y_train, spw)

    val_score_jit = model_jit.predict_proba(val_df[JIT_FEATURES])[:, 1]
    test_score_jit = model_jit.predict_proba(test_df[JIT_FEATURES])[:, 1]
    val_score_cb = model_codebert.predict_proba(val_df[PCA_COLS])[:, 1]
    test_score_cb = model_codebert.predict_proba(test_df[PCA_COLS])[:, 1]
    val_score_4a = model_4a.predict_proba(val_df[early_features])[:, 1]
    test_score_4a = model_4a.predict_proba(test_df[early_features])[:, 1]
    val_score_4b1 = model_4b1.predict_proba(val_df[features_4b1])[:, 1]
    test_score_4b1 = model_4b1.predict_proba(test_df[features_4b1])[:, 1]
    val_score_4b2 = model_4b2.predict_proba(val_df[features_4b2])[:, 1]
    test_score_4b2 = model_4b2.predict_proba(test_df[features_4b2])[:, 1]
    val_score_4c = model_4c.predict_proba(val_df[top25_mi])[:, 1]
    test_score_4c = model_4c.predict_proba(test_df[top25_mi])[:, 1]

    val_meta = np.column_stack([val_score_jit, val_score_cb])
    test_meta = np.column_stack([test_score_jit, test_score_cb])
    meta = LogisticRegression(solver="lbfgs", random_state=SEED)
    meta.fit(val_meta, y_val)
    val_score_4d = meta.predict_proba(val_meta)[:, 1]
    test_score_4d = meta.predict_proba(test_meta)[:, 1]

    best_alpha, best_blend_f1 = 0.5, -1.0
    for alpha in np.linspace(0.1, 0.9, 17):
        blend_val = alpha * val_score_jit + (1 - alpha) * val_score_cb
        _, blend_metrics = find_optimal_threshold(y_val, blend_val, np.linspace(0.01, 0.99, 99))
        if blend_metrics["f1"] > best_blend_f1:
            best_alpha, best_blend_f1 = float(alpha), blend_metrics["f1"]
    val_score_4e = best_alpha * val_score_jit + (1 - best_alpha) * val_score_cb
    test_score_4e = best_alpha * test_score_jit + (1 - best_alpha) * test_score_cb

    variants = {
        "JIT_Only": (val_score_jit, test_score_jit, "JIT metrics only"),
        "CodeBERT_Only": (val_score_cb, test_score_cb, "CodeBERT PCA features only"),
        "4a_Early_Fusion_All": (val_score_4a, test_score_4a, "13 JIT + 384 PCA"),
        "4b1_Early_Fusion_Top25_PCA": (val_score_4b1, test_score_4b1, "13 JIT + 25 PCA"),
        "4b2_Early_Fusion_Top50_PCA": (val_score_4b2, test_score_4b2, "13 JIT + 50 PCA"),
        "4c_MI_Top25_Selected": (val_score_4c, test_score_4c, "Top-25 mutual information features"),
        "4d_Stacking_Ensemble": (val_score_4d, test_score_4d, "Validation-trained logistic stacking"),
        "4e_Probability_Blend": (val_score_4e, test_score_4e, f"{best_alpha:.2f} JIT + {1-best_alpha:.2f} CodeBERT blend"),
    }

    validation_records, test_records = [], []
    predictions = test_df[["commit_id", "project", "author_date"]].copy()
    predictions["buggy_true"] = y_test
    for name, (val_score, test_score, description) in variants.items():
        threshold, val_metrics = find_optimal_threshold(y_val, val_score, np.linspace(0.01, 0.99, 99))
        test_metrics = compute_metrics(y_test, test_score, threshold)
        validation_records.append({"variant": name, "description": description, **{f"val_{key}": value for key, value in val_metrics.items()}})
        test_records.append({"variant": name, "description": description, **{f"test_{key}": value for key, value in test_metrics.items()}})
        predictions[f"score_{name}"] = test_score
        predictions[f"pred_{name}"] = (test_score >= threshold).astype(int)

    pd.DataFrame(validation_records).to_csv(OUTPUT_DIR / "tables" / "validation_fusion_comparison.csv", index=False)
    pd.DataFrame(test_records).to_csv(OUTPUT_DIR / "tables" / "final_test_fusion_comparison.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "predictions" / "final_test_predictions.csv", index=False)
    for name, model in {"jit": model_jit, "codebert": model_codebert, "4a": model_4a, "4b1": model_4b1, "4b2": model_4b2, "4c": model_4c, "stacking": meta}.items():
        import joblib
        joblib.dump(model, OUTPUT_DIR / "models" / f"{name}.joblib")

    summary = {
        "experiment": "multilanguage_experiment_4_controlled_v1",
        "dataset_path": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "features_jit": JIT_FEATURES,
        "features_codebert": len(PCA_COLS),
        "train_rows": len(train_df), "validation_rows": len(val_df), "test_rows": len(test_df),
        "scale_pos_weight": spw, "best_blend_alpha": best_alpha,
        "train_target_distribution": y_train.value_counts().sort_index().to_dict(),
        "validation_target_distribution": pd.Series(y_val).value_counts().sort_index().to_dict(),
        "test_target_distribution": pd.Series(y_test).value_counts().sort_index().to_dict(),
        "validation_results": validation_records, "final_test_results": test_records,
    }
    (OUTPUT_DIR / "metrics" / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else value), encoding="utf-8")
    logger.info("Completed Exp4: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
