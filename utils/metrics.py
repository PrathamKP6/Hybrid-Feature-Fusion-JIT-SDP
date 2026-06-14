"""Model evaluation metrics for the Hybrid Feature Fusion project.

This module provides convenience functions to compute standard binary
classification metrics and return them in a dictionary.
"""
from __future__ import annotations

from typing import Dict, Sequence

import logging
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[float],
    pos_label: int = 1,
) -> Dict[str, float]:
    """Compute evaluation metrics for binary classification.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        y_score: Score or probability for the positive class.
        pos_label: Positive class label.

    Returns:
        Dictionary with accuracy, precision, recall, f1, roc_auc, pr_auc, mcc,
        and confusion matrix elements.
    """
    try:
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)
        mcc = matthews_corrcoef(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except Exception as exc:
        logger.exception("Failed to compute classification metrics: %s", exc)
        raise

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "mcc": float(mcc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


__all__ = ["compute_classification_metrics"]
