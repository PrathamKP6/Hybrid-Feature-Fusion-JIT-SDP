"""Data cleaning utilities for the Hybrid Feature Fusion project.

This module provides functions to perform basic cleaning operations on the
commit dataset used for Experiment 1 (JIT features only).

Responsibilities:
- Remove duplicate `commit_id` rows.
- Remove rows with missing target labels.
- Handle missing values in JIT feature columns (imputation strategies).
- Log all cleaning operations and return a concise cleaning report.

This module intentionally does NOT perform sampling or train/test splitting.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import logging
import pandas as pd

from preprocessing.load_data import MissingColumnsError

logger = logging.getLogger(__name__)


class DataCleanError(Exception):
    """Base exception for data cleaning errors."""


def clean_commits_df(
    df: pd.DataFrame,
    jit_feature_columns: Optional[Sequence[str]] = None,
    target_column: str = "buggy",
    id_column: str = "commit_id",
    drop_duplicates_keep: str = "first",
    impute_strategy: str = "median",
    per_column_fill: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean a commits DataFrame for Experiment 1 (JIT features only).

    The function performs the following operations (in order):
    1. Validate that required columns exist in ``df``.
    2. Remove duplicate rows based on ``id_column``.
    3. Drop rows with missing values in ``target_column``.
    4. Impute missing values in the specified ``jit_feature_columns``.

    The function returns the cleaned DataFrame and a report dictionary with
    counts and details describing the performed operations.

    Args:
        df: Input DataFrame to clean. This is not modified in-place; a copy is
            returned.
        jit_feature_columns: Sequence of column names corresponding to JIT
            features. If ``None``, no JIT-specific imputation is performed.
        target_column: Name of the target/label column. Rows with missing
            values in this column will be removed.
        id_column: Column used to detect duplicate commits.
        drop_duplicates_keep: Passed to ``DataFrame.drop_duplicates``'s
            ``keep`` argument (e.g., "first", "last", or False).
        impute_strategy: Strategy used when ``per_column_fill`` does not
            provide a value for a column. Supported: "median", "mean",
            "mode", "zero".
        per_column_fill: Optional dict mapping column -> fill value. If a
            column appears here, its value is used to fill missing entries
            instead of the generic strategy.

    Returns:
        A tuple ``(cleaned_df, report)`` where ``cleaned_df`` is the cleaned
        pandas DataFrame and ``report`` is a dict summarizing operations.

    Raises:
        MissingColumnsError: If required columns (``id_column`` or
            ``target_column``) are not present in ``df``.
        DataCleanError: For unexpected errors during cleaning.
    """
    # Work on a copy to avoid mutating caller's DataFrame
    df = df.copy()

    # Validate required columns
    required = [id_column, target_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error("Missing required columns for cleaning: %s", missing)
        raise MissingColumnsError(missing)

    if jit_feature_columns is None:
        jit_feature_columns = []
    else:
        # ensure columns exist
        missing_jit = [c for c in jit_feature_columns if c not in df.columns]
        if missing_jit:
            logger.error("Missing specified JIT feature columns: %s", missing_jit)
            raise MissingColumnsError(missing_jit)

    report: Dict[str, Any] = {
        "initial_rows": int(len(df)),
        "duplicates_removed": 0,
        "rows_removed_missing_target": 0,
        "missing_before": {},
        "missing_after": {},
        "imputed_values": {},
    }

    try:
        # 1) Remove duplicate commit ids
        before = len(df)
        df = df.drop_duplicates(subset=[id_column], keep=drop_duplicates_keep)
        after = len(df)
        removed_dup = before - after
        report["duplicates_removed"] = int(removed_dup)
        logger.info("Removed %d duplicate rows based on %s", removed_dup, id_column)

        # 2) Remove rows with missing target labels
        before = len(df)
        df = df.dropna(subset=[target_column])
        after = len(df)
        removed_target = before - after
        report["rows_removed_missing_target"] = int(removed_target)
        logger.info("Removed %d rows with missing target '%s'", removed_target, target_column)

        # 3) Missing values report before imputation
        missing_before = {col: int(df[col].isna().sum()) for col in jit_feature_columns}
        report["missing_before"] = missing_before
        logger.debug("Missing values before imputation: %s", missing_before)

        # 4) Impute missing values in JIT features
        imputed_values: Dict[str, Any] = {}
        for col in jit_feature_columns:
            missing_count = int(df[col].isna().sum())
            if missing_count == 0:
                continue

            # Use per-column fill if provided
            if per_column_fill and col in per_column_fill:
                fill_value = per_column_fill[col]
                df[col] = df[col].fillna(fill_value)
                imputed_values[col] = fill_value
                logger.info("Imputed %d missing values in '%s' using per_column_fill=%s", missing_count, col, fill_value)
                continue

            # Generic strategies
            try:
                if impute_strategy == "median":
                    fill_value = df[col].median()
                elif impute_strategy == "mean":
                    fill_value = df[col].mean()
                elif impute_strategy == "mode":
                    modes = df[col].mode()
                    fill_value = modes.iloc[0] if not modes.empty else 0
                elif impute_strategy == "zero":
                    fill_value = 0
                else:
                    logger.warning("Unknown impute_strategy '%s', defaulting to median", impute_strategy)
                    fill_value = df[col].median()

                # If fill_value is NaN (e.g., column entirely NaN), fall back to 0
                if pd.isna(fill_value):
                    logger.warning("Computed fill value for column '%s' is NaN; falling back to 0", col)
                    fill_value = 0

                df[col] = df[col].fillna(fill_value)
                imputed_values[col] = fill_value
                logger.info("Imputed %d missing values in '%s' using strategy '%s' (value=%s)", missing_count, col, impute_strategy, fill_value)
            except Exception as exc:
                logger.exception("Failed to impute column '%s': %s", col, exc)
                raise DataCleanError(f"Failed to impute column '{col}': {exc}") from exc

        # 5) Missing values report after imputation
        missing_after = {col: int(df[col].isna().sum()) for col in jit_feature_columns}
        report["missing_after"] = missing_after
        report["imputed_values"] = imputed_values
        logger.debug("Missing values after imputation: %s", missing_after)

        report["final_rows"] = int(len(df))

        return df, report

    except Exception as exc:
        logger.exception("Unexpected error during cleaning: %s", exc)
        raise DataCleanError(f"Unexpected error during cleaning: {exc}") from exc


__all__ = ["clean_commits_df", "DataCleanError"]
