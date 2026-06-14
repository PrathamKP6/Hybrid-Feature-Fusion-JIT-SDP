"""XGBoost training utilities for the Hybrid Feature Fusion project.

This module trains an XGBClassifier for Experiment 1 and persists the trained
model with joblib.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import logging
import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_path: str | Path,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    scale_pos_weight: float | None = None,
    random_state: int = 42,
) -> Tuple[ClassifierMixin, pd.Series, pd.Series]:
    """Train an XGBClassifier and save the trained model.

    Args:
        X_train: Training features.
        y_train: Training target labels.
        X_test: Test features.
        y_test: Test target labels.
        model_path: Path where the trained model will be saved with joblib.
        n_estimators: Number of boosting rounds.
        learning_rate: Learning rate for XGBoost.
        scale_pos_weight: Value to balance positive and negative classes.
        random_state: Random seed for reproducibility.

    Returns:
        A tuple ``(model, y_pred, y_score)`` where ``y_pred`` are predicted
        class labels and ``y_score`` are predicted probabilities for the
        positive class.
    """
    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state,
    )

    logger.info(
        "Training XGBClassifier with n_estimators=%d, lr=%s, scale_pos_weight=%s",
        n_estimators,
        learning_rate,
        scale_pos_weight,
    )
    model.fit(X_train, y_train)

    logger.info("XGBClassifier training complete")
    y_pred = pd.Series(model.predict(X_test), index=X_test.index)
    y_score = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Saved XGBClassifier model to %s", model_path)

    return model, y_pred, y_score


__all__ = ["train_xgboost"]
