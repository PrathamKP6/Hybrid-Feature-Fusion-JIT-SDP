"""Temporal splitting utilities for the Hybrid Feature Fusion project.

This module provides helpers to split a commit dataset chronologically into a
training and test set without shuffling. The split is based on commit
timestamps and preserves temporal ordering for evaluation.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import logging
import pandas as pd

logger = logging.getLogger(__name__)


class TemporalSplitError(Exception):
    """Raised when temporal splitting cannot be completed."""


def temporal_train_test_split(
    df: pd.DataFrame,
    timestamp_column: str = "author_date",
    target_column: str = "buggy",
    feature_columns: Optional[Sequence[str]] = None,
    train_size: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split commits chronologically into training and testing sets.

    Args:
        df: Input DataFrame containing commit records.
        timestamp_column: Column containing commit timestamps.
        target_column: Column name for the target label.
        feature_columns: Columns to use as features. If ``None``, all columns
            except the target and timestamp columns will be used.
        train_size: Fraction of rows to use for training (default 0.8).

    Returns:
        A tuple ``(X_train, X_test, y_train, y_test)``.

    Raises:
        TemporalSplitError: If required columns are missing or parameters are invalid.
    """
    if timestamp_column not in df.columns:
        logger.error("Timestamp column '%s' not found", timestamp_column)
        raise TemporalSplitError(f"Timestamp column '{timestamp_column}' not found")
    if target_column not in df.columns:
        logger.error("Target column '%s' not found", target_column)
        raise TemporalSplitError(f"Target column '{target_column}' not found")
    if not 0.0 < train_size < 1.0:
        logger.error("train_size must be between 0.0 and 1.0, got %s", train_size)
        raise TemporalSplitError("train_size must be between 0.0 and 1.0")

    df_sorted = df.sort_values(by=timestamp_column, ascending=True).reset_index(drop=True)
    total = len(df_sorted)
    if total == 0:
        logger.error("Input DataFrame is empty")
        raise TemporalSplitError("Input DataFrame is empty")

    split_index = int(total * train_size)
    X_columns = [col for col in df_sorted.columns if col != target_column and col != timestamp_column]
    if feature_columns is not None:
        missing_features = [col for col in feature_columns if col not in df_sorted.columns]
        if missing_features:
            logger.error("Feature columns not found: %s", missing_features)
            raise TemporalSplitError(f"Feature columns not found: {missing_features}")
        X_columns = list(feature_columns)

    X_train = df_sorted.loc[: split_index - 1, X_columns].copy()
    X_test = df_sorted.loc[split_index:, X_columns].copy()
    y_train = df_sorted.loc[: split_index - 1, target_column].copy()
    y_test = df_sorted.loc[split_index:, target_column].copy()

    logger.info(
        "Temporal split: %d train rows, %d test rows, split index %d of %d",
        len(X_train),
        len(X_test),
        split_index,
        total,
    )
    return X_train, X_test, y_train, y_test


__all__ = ["temporal_train_test_split", "TemporalSplitError"]
