
"""Inference utilities for the frozen Experiment 7F models."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_DIR = ROOT / "results" / "experiment_7f_probability_blending"
PCA_DIR = ROOT / "results" / "pca"
LLM_DIR = ROOT / "data" / "llm_experiment_dataset"

JIT_FEATURES = [
    "la", "ld", "nf", "ns", "nd", "ent",
    "ndev", "age", "nuc", "aexp", "arexp", "asexp",
]

LLM_CATEGORICAL = [
    "intent", "change", "risk", "complexity",
    "scope", "test", "security",
]

LLM_NUMERICAL = [
    "intent_confidence", "intent_margin",
    "change_confidence", "change_margin",
    "risk_confidence", "risk_margin",
    "complexity_confidence", "complexity_margin",
    "scope_confidence", "scope_margin",
    "test_confidence", "test_margin",
    "security_confidence", "security_margin",
]

THRESHOLD = 0.50


def parse_embedding(value):
    """Parse a stored 384-dimensional PCA embedding."""
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=np.float32)
    else:
        text = str(value).strip()
        try:
            arr = np.asarray(json.loads(text), dtype=np.float32)
        except (ValueError, TypeError):
            arr = np.fromstring(
                text.strip("[]"), sep=",", dtype=np.float32
            )

    arr = arr.reshape(-1)
    if arr.size != 384:
        raise ValueError(
            f"Expected 384 embedding values, got {arr.size}"
        )
    return arr


def load_models():
    model_dir = EXPERIMENT_DIR / "models"
    return {
        "random_forest": joblib.load(
            model_dir / "random_forest.joblib"
        ),
        "xgboost": joblib.load(model_dir / "xgboost.joblib"),
        "lightgbm": joblib.load(model_dir / "lightgbm.joblib"),
    }


def load_blend_weights():
    with (EXPERIMENT_DIR / "metrics.json").open(
        "r", encoding="utf-8"
    ) as file:
        metrics = json.load(file)

    weights = metrics["selected_weights"]
    return {
        "random_forest": float(weights["random_forest"]),
        "xgboost": float(weights["xgboost"]),
        "lightgbm": float(weights["lightgbm"]),
    }


def load_features(split="test"):
    """Build 447 features with LLM preprocessing fitted on TRAIN only."""
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'")

    train_llm = pd.read_csv(LLM_DIR / "train.csv")
    split_pca = pd.read_csv(PCA_DIR / f"{split}_pca384.csv")
    split_llm = pd.read_csv(LLM_DIR / f"{split}.csv")

    if split_pca["commit_id"].duplicated().any():
        raise ValueError("Duplicate commit IDs in PCA data")
    if split_llm["commit_id"].duplicated().any():
        raise ValueError("Duplicate commit IDs in LLM data")

    llm_indexed = split_llm.set_index("commit_id")
    missing_ids = split_pca.loc[
        ~split_pca["commit_id"].isin(llm_indexed.index), "commit_id"
    ]
    if not missing_ids.empty:
        raise ValueError(
            f"{len(missing_ids)} PCA commits lack LLM features"
        )

    # Align the LLM rows to the PCA dataset by commit ID.
    split_llm = llm_indexed.loc[
        split_pca["commit_id"]
    ].reset_index()

    if not np.array_equal(
        split_pca["commit_id"].astype(str).to_numpy(),
        split_llm["commit_id"].astype(str).to_numpy(),
    ):
        raise ValueError("PCA/LLM commit alignment failed")

    # Fit the encoder and imputer on TRAIN only.
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )
    imputer = SimpleImputer(strategy="median")

    encoder.fit(
        train_llm[LLM_CATEGORICAL].fillna("UNKNOWN").astype(str)
    )
    split_cat = encoder.transform(
        split_llm[LLM_CATEGORICAL].fillna("UNKNOWN").astype(str)
    )

    imputer.fit(train_llm[LLM_NUMERICAL])
    split_num = imputer.transform(split_llm[LLM_NUMERICAL])

    llm_features = np.hstack(
        [split_cat, split_num]
    ).astype(np.float32)

    jit_features = split_pca[JIT_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float32)

    codebert_features = np.vstack([
        parse_embedding(value)
        for value in split_pca["embedding"]
    ]).astype(np.float32)

    X = np.hstack([
        jit_features, codebert_features, llm_features
    ]).astype(np.float32)

    if X.shape[1] != 447:
        raise ValueError(f"Expected 447 features, got {X.shape[1]}")

    if not np.isfinite(X).all():
        raise ValueError(
            "Feature matrix contains NaN or infinite values. "
            "Stopping rather than silently changing preprocessing."
        )

    return split_pca, X


def predict_split(split="test"):
    """Predict a canonical split using the saved models."""
    frame, X = load_features(split)
    models = load_models()
    weights = load_blend_weights()

    probabilities = {
        name: model.predict_proba(X)[:, 1]
        for name, model in models.items()
    }

    blended = sum(
        weights[name] * probabilities[name]
        for name in weights
    )

    return pd.DataFrame({
        "commit_id": frame["commit_id"].astype(str).to_numpy(),
        "rf_probability": probabilities["random_forest"],
        "xgb_probability": probabilities["xgboost"],
        "lightgbm_probability": probabilities["lightgbm"],
        "blend_probability": blended,
        "blend_prediction": (blended >= THRESHOLD).astype(int),
    })
