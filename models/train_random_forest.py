"""Random Forest training utilities for the Hybrid Feature Fusion project.

This module trains a scikit-learn RandomForestClassifier for Experiment 1 and
persists the fitted model with joblib.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import logging
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import ClassifierMixin

logger = logging.getLogger(__name__)


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_path: str | Path,
    n_estimators: int = 100,
    random_state: int = 42,
    class_weight: str = "balanced",
) -> Tuple[ClassifierMixin, pd.Series, pd.Series]:
    """Train a RandomForestClassifier and save the trained model.

    Args:
        X_train: Training features.
        y_train: Training target labels.
        X_test: Test features.
        y_test: Test target labels.
        model_path: Path where the trained model will be saved with joblib.
        n_estimators: Number of trees in the forest.
        random_state: Random seed for reproducibility.
        class_weight: Class weight strategy for imbalance handling.

    Returns:
        A tuple of ``(model, y_pred, y_score)`` where ``y_pred`` are predicted
        class labels and ``y_score`` are predicted probabilities for the positive
        class.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight=class_weight,
    )

    logger.info("Training RandomForestClassifier with n_estimators=%d, class_weight=%s", n_estimators, class_weight)
    model.fit(X_train, y_train)

    logger.info("RandomForestClassifier training complete")
    y_pred = pd.Series(model.predict(X_test), index=X_test.index)

    if hasattr(model, "predict_proba"):
        logger.info("Computing predicted probabilities for the positive class")
        y_score = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)
    else:
        logger.warning("Model does not support predict_proba; using decision function instead if available")
        if hasattr(model, "decision_function"):
            y_score = pd.Series(model.decision_function(X_test), index=X_test.index)
        else:
            raise ValueError("Model does not support probability or decision function outputs")

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Saved RandomForest model to %s", model_path)

    return model, y_pred, y_score


__all__ = ["train_random_forest"]
