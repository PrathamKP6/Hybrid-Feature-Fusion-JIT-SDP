"""Shared implementation for the final full-dataset experiments."""
from __future__ import annotations

import json
import ast
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "final_multilanguage_59996_pca384.csv"
SPLIT_DIR = PROJECT_ROOT / "results" / "final_shared_split"
JIT_FEATURES = ["la", "ld", "nf", "ns", "nd", "ent", "ndev", "age", "nuc", "aexp", "arexp", "asexp"]
PCA_FEATURES = [f"pca_{index}" for index in range(1, 385)]
REQUIRED_BASE_COLUMNS = ["commit_id", "author_date", "buggy", "project", *JIT_FEATURES]
SEED = 42


def parse_dates(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() >= max(1, int(len(series) * 0.1)):
        maximum = float(numeric.abs().max())
        unit = "ns" if maximum > 10**15 else "ms" if maximum > 10**12 else "s"
        parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")
        numeric_mask = numeric.notna()
        parsed.loc[numeric_mask] = pd.to_datetime(numeric.loc[numeric_mask], unit=unit, errors="coerce", utc=True)
        parsed.loc[~numeric_mask] = pd.to_datetime(series.loc[~numeric_mask], format="mixed", errors="coerce", utc=True)
        return parsed
    return pd.to_datetime(series, format="mixed", errors="coerce", utc=True)


def validate_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Full PCA384 dataset not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    missing = [column for column in REQUIRED_BASE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")
    explicit_pca = all(column in df.columns for column in PCA_FEATURES)
    if not explicit_pca:
        if path.name != "final_multilanguage_59996_pca384.csv":
            raise ValueError("Serialized embeddings are permitted only for the known final_multilanguage_59996_pca384.csv artifact")
        if "embedding" not in df.columns:
            raise ValueError("Missing explicit PCA columns and serialized embedding column")
        parsed_embeddings: list[np.ndarray] = []
        for row_number, value in enumerate(df["embedding"], start=2):
            if pd.isna(value):
                raise ValueError(f"Embedding is null at CSV row {row_number}")
            try:
                parsed = ast.literal_eval(value) if isinstance(value, str) else value
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"Embedding parsing failed at CSV row {row_number}: {exc}") from exc
            try:
                vector = np.asarray(parsed, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Embedding contains non-numeric values at CSV row {row_number}: {exc}") from exc
            if vector.ndim != 1 or vector.size != 384:
                raise ValueError(f"Embedding at CSV row {row_number} has dimension {vector.size}; expected exactly 384")
            if not np.isfinite(vector).all():
                raise ValueError(f"Embedding at CSV row {row_number} contains non-finite values")
            parsed_embeddings.append(vector)
        if len(parsed_embeddings) != len(df):
            raise ValueError(f"Validated {len(parsed_embeddings)} embeddings for {len(df)} rows")
        df = pd.concat(
            [df.drop(columns=["embedding"]), pd.DataFrame(np.vstack(parsed_embeddings), columns=PCA_FEATURES)],
            axis=1,
        )
    else:
        if "embedding" in df.columns:
            df = df.drop(columns=["embedding"])
    if df["commit_id"].isna().any():
        raise ValueError("commit_id must be non-null before splitting")
    df = df.drop_duplicates(subset=["commit_id"], keep="first").reset_index(drop=True)
    df["author_date"] = parse_dates(df["author_date"])
    if df["author_date"].isna().any():
        raise ValueError("author_date contains invalid or missing values")
    normalized_labels = df["buggy"].astype(str).str.strip().str.lower().map({"0": 0, "1": 1, "false": 0, "true": 1})
    if normalized_labels.isna().any() or set(normalized_labels.unique()) != {0, 1}:
        found_labels = sorted(df["buggy"].astype(str).drop_duplicates().tolist())
        raise ValueError(f"buggy must contain only binary 0/1 or True/False labels; found {found_labels}")
    df["buggy"] = normalized_labels.astype(int)
    for column in [*JIT_FEATURES, *PCA_FEATURES]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if np.isinf(df[column].dropna()).any():
            raise ValueError(f"Feature {column} contains infinite values")
    for column in PCA_FEATURES:
        if df[column].isna().any():
            raise ValueError(f"PCA feature {column} contains missing values")
    if len(df) != 59_996:
        raise ValueError(f"Expected exactly 59,996 logical rows, found {len(df)}")
    if len(df) < 50_000:
        raise ValueError(f"Expected the full approximately 60K dataset, found {len(df)} rows")
    return df


def _distribution(frame: pd.DataFrame) -> dict[str, Any]:
    counts = frame["buggy"].value_counts().sort_index().to_dict()
    return {"rows": int(len(frame)), "buggy_0": int(counts.get(0, 0)), "buggy_1": int(counts.get(1, 0)), "buggy_prevalence": float(frame["buggy"].mean())}


def _date_range(frame: pd.DataFrame) -> list[str]:
    return [frame["author_date"].min().isoformat(), frame["author_date"].max().isoformat()]


def create_or_load_split(df: pd.DataFrame, force: bool = False) -> dict[str, pd.DataFrame]:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = SPLIT_DIR / "split_manifest.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        by_id = df.set_index("commit_id", drop=False)
        splits: dict[str, pd.DataFrame] = {}
        for name in ("train", "validation", "test"):
            ids = pd.read_csv(SPLIT_DIR / f"{name}_commit_ids.csv")["commit_id"]
            if ids.duplicated().any() or not set(ids).issubset(set(by_id.index)):
                raise ValueError(f"Invalid or stale {name} split manifest")
            splits[name] = by_id.loc[ids].reset_index(drop=True)
        split_ids = set().union(*(set(frame["commit_id"]) for frame in splits.values()))
        if len(split_ids) != len(df) or split_ids != set(df["commit_id"]):
            raise ValueError("Split manifest does not cover exactly the validated dataset")
        if set(splits["train"]["commit_id"]) & set(splits["validation"]["commit_id"]):
            raise ValueError("Train and validation split overlap")
        if set(splits["train"]["commit_id"]) & set(splits["test"]["commit_id"]):
            raise ValueError("Train and test split overlap")
        if set(splits["validation"]["commit_id"]) & set(splits["test"]["commit_id"]):
            raise ValueError("Validation and test split overlap")
        if manifest.get("dataset_rows") != len(df):
            raise ValueError("Split manifest was created for a different dataset row count")
        return splits

    ordered = df.sort_values("author_date", kind="mergesort").reset_index(drop=True)
    train_end = int(len(ordered) * 0.60)
    validation_end = int(len(ordered) * 0.80)
    splits = {"train": ordered.iloc[:train_end].copy(), "validation": ordered.iloc[train_end:validation_end].copy(), "test": ordered.iloc[validation_end:].copy()}
    manifest: dict[str, Any] = {
        "dataset_path": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "dataset_rows": len(ordered),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "method": "stable chronological 60/20/20 split after duplicate validation",
        "splits": {},
    }
    for name, frame in splits.items():
        frame[["commit_id"]].to_csv(SPLIT_DIR / f"{name}_commit_ids.csv", index=False)
        manifest["splits"][name] = {**_distribution(frame), "date_range": _date_range(frame)}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return splits


def prepare_features(splits: dict[str, pd.DataFrame], features: Sequence[str]) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    train = splits["train"]
    medians = {feature: float(train[feature].median()) for feature in JIT_FEATURES if train[feature].isna().any()}
    prepared: dict[str, pd.DataFrame] = {}
    for name, frame in splits.items():
        values = frame[list(features)].copy()
        for feature, median in medians.items():
            if feature in values:
                values[feature] = values[feature].fillna(median)
        if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite feature values remain in {name} after train-only preprocessing")
        prepared[name] = values
    return prepared, medians


def metrics(y_true: pd.Series | np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, labels, labels=[0, 1]).ravel()
    return {"threshold": float(threshold), "roc_auc": float(roc_auc_score(y_true, probabilities)), "pr_auc": float(average_precision_score(y_true, probabilities)), "precision": float(precision_score(y_true, labels, zero_division=0)), "recall": float(recall_score(y_true, labels, zero_division=0)), "f1": float(f1_score(y_true, labels, zero_division=0)), "mcc": float(matthews_corrcoef(y_true, labels)), "accuracy": float(accuracy_score(y_true, labels)), "brier": float(brier_score_loss(y_true, probabilities)), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "predicted_positive_count": int(labels.sum()), "predicted_positive_rate": float(labels.mean())}


def select_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, dict[str, Any]]:
    candidates = np.round(np.arange(0.01, 1.00, 0.01), 2)
    results = [(float(threshold), metrics(y_true, probabilities, float(threshold))) for threshold in candidates]
    return max(results, key=lambda item: item[1]["f1"])


def model_parameters(scale_pos_weight: float) -> tuple[dict[str, Any], dict[str, Any]]:
    return ({"n_estimators": 100, "class_weight": "balanced", "random_state": SEED, "n_jobs": -1}, {"objective": "binary:logistic", "n_estimators": 100, "learning_rate": 0.1, "eval_metric": "logloss", "scale_pos_weight": scale_pos_weight, "random_state": SEED, "n_jobs": -1})


def run_experiment(name: str, features: Sequence[str], output_dir: Path, smoke: bool = False) -> None:
    import xgboost
    from xgboost import XGBClassifier

    df = validate_dataset()
    splits = create_or_load_split(df)
    prepared, medians = prepare_features(splits, features)
    train, validation, test = splits["train"], splits["validation"], splits["test"]
    y_train, y_validation, y_test = train["buggy"], validation["buggy"], test["buggy"]
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    rf_params, xgb_params = model_parameters(scale_pos_weight)
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("metrics", "predictions", "models"):
        (output_dir / subdir).mkdir(exist_ok=True)
    training_rows = min(256, len(train)) if smoke else len(train)
    rf = RandomForestClassifier(**rf_params).fit(prepared["train"].iloc[:training_rows], y_train.iloc[:training_rows])
    xgb = XGBClassifier(**xgb_params).fit(prepared["train"].iloc[:training_rows], y_train.iloc[:training_rows])
    validation_input = prepared["validation"].iloc[:64] if smoke else prepared["validation"]
    test_input = prepared["test"].iloc[:64] if smoke else prepared["test"]
    validation_labels = y_validation.iloc[:len(validation_input)]
    test_labels = y_test.iloc[:len(test_input)]
    records: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for model_name, model in (("random_forest", rf), ("xgboost", xgb)):
        validation_prob = model.predict_proba(validation_input)[:, 1]
        test_prob = model.predict_proba(test_input)[:, 1]
        threshold, validation_metrics = select_threshold(validation_labels, validation_prob)
        test_metrics = metrics(test_labels, test_prob, threshold)
        records.append({"model": model_name, "validation": validation_metrics, "final_test": test_metrics})
        joblib.dump(model, output_dir / "models" / f"{model_name}.joblib")
        if hasattr(model, "feature_importances_"):
            pd.DataFrame({"feature": list(features), "importance": model.feature_importances_}).sort_values("importance", ascending=False).to_csv(output_dir / "metrics" / f"{model_name}_feature_importance.csv", index=False)
        prediction_frame = test.iloc[:len(test_input)][["commit_id", "author_date", "project", "buggy"]].copy()
        prediction_frame.rename(columns={"buggy": "actual_buggy"}, inplace=True)
        prediction_frame["predicted_probability"] = test_prob
        prediction_frame["predicted_label"] = (test_prob >= threshold).astype(int)
        prediction_frame["model"] = model_name
        prediction_frame["threshold"] = threshold
        prediction_frames.append(prediction_frame)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / "predictions" / "final_test_predictions.csv", index=False)
    split_summary = json.loads((SPLIT_DIR / "split_manifest.json").read_text(encoding="utf-8"))
    configuration = {"experiment": name, "dataset_path": str(DATA_PATH.relative_to(PROJECT_ROOT)), "dataset_rows": len(df), "features": list(features), "feature_count": len(features), "split_manifest": str(SPLIT_DIR.relative_to(PROJECT_ROOT)), "train_only_jit_medians": medians, "scale_pos_weight": scale_pos_weight, "random_forest": rf_params, "xgboost": xgb_params, "created_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version, "platform": platform.platform(), "sklearn": sklearn.__version__, "xgboost_version": xgboost.__version__}
    (output_dir / "metrics" / "validation_metrics.json").write_text(json.dumps({record["model"]: record["validation"] for record in records}, indent=2), encoding="utf-8")
    (output_dir / "metrics" / "final_test_metrics.json").write_text(json.dumps({record["model"]: record["final_test"] for record in records}, indent=2), encoding="utf-8")
    (output_dir / "metrics" / "configuration.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    (output_dir / "metrics" / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")


def smoke_validate() -> dict[str, Any]:
    df = validate_dataset()
    splits = create_or_load_split(df)
    return {name: {**_distribution(frame), "date_range": _date_range(frame)} for name, frame in splits.items()}