"""Sampling utilities for the Hybrid Feature Fusion project.

This module handles temporal filtering and stratified sampling required for the
initial development dataset. It preserves original timestamps and creates an
approximate subset of commits for Experiment 1.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import logging
import pandas as pd

logger = logging.getLogger(__name__)


class SamplingError(Exception):
    """Base exception for sampling-related failures."""


def filter_commits_by_date(
    df: pd.DataFrame,
    timestamp_column: str = "author_date",
    start_year: int = 2018,
    end_year: int = 2026,
) -> pd.DataFrame:
    """Filter commits to a date range inclusive of both start and end year.

    Args:
        df: Input DataFrame containing commit data.
        timestamp_column: Name of the timestamp column.
        start_year: Minimum year to keep, inclusive.
        end_year: Maximum year to keep, inclusive.

    Returns:
        A filtered DataFrame containing commits whose timestamp year falls
        between ``start_year`` and ``end_year``.

    Raises:
        SamplingError: If the timestamp column is missing or cannot be parsed.
    """
    if timestamp_column not in df.columns:
        logger.error("Timestamp column '%s' not found in DataFrame", timestamp_column)
        raise SamplingError(f"Timestamp column '{timestamp_column}' not found in DataFrame")

    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
        try:
            df = df.copy()
            df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors="raise")
        except Exception as exc:
            logger.exception("Failed to parse timestamps in '%s': %s", timestamp_column, exc)
            raise SamplingError(f"Failed to parse timestamps in '{timestamp_column}': {exc}") from exc

    before_rows = len(df)
    year = df[timestamp_column].dt.year
    filtered = df[(year >= start_year) & (year <= end_year)].copy()
    after_rows = len(filtered)
    logger.info(
        "Filtered commits by date range %d-%d: %d -> %d rows",
        start_year,
        end_year,
        before_rows,
        after_rows,
    )
    return filtered


def stratified_sample(
    df: pd.DataFrame,
    group_columns: Sequence[str] = ("project", "buggy"),
    target_size: int = 10000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Perform stratified sampling by language and bug label.

    The function preserves the original timestamps and returns an approximate
    subset of commits with the same group distribution as the input.

    Args:
        df: Input DataFrame to sample.
        group_columns: Columns used for stratification.
        target_size: Approximate number of samples to return.
        random_state: Seed for reproducibility.

    Returns:
        A sampled DataFrame with approximately ``target_size`` rows.

    Raises:
        SamplingError: If required stratification columns are missing.
    """
    missing = [col for col in group_columns if col not in df.columns]
    if missing:
        logger.error("Missing required stratification columns: %s", missing)
        raise SamplingError(f"Missing required stratification columns: {missing}")

    group_counts = df.groupby(list(group_columns)).size().reset_index(name="count")
    total = len(df)
    if total == 0:
        logger.error("Cannot sample from an empty DataFrame")
        raise SamplingError("Cannot sample from an empty DataFrame")

    logger.info("Performing stratified sampling from %d rows using groups %s", total, list(group_columns))

    # Calculate approximate sample sizes per group using proportional allocation.
    group_counts["target_count"] = (group_counts["count"] / total * target_size).round().astype(int)

    # Ensure at least one sample for every group with data.
    group_counts["target_count"] = group_counts["target_count"].clip(lower=1)

    sampled_frames: List[pd.DataFrame] = []
    actual_sampled = 0
    for _, row in group_counts.iterrows():
        group_values = [row[col] for col in group_columns]
        group_df = df
        for col, value in zip(group_columns, group_values):
            group_df = group_df[group_df[col] == value]

        desired = int(row["target_count"])
        available = len(group_df)
        if desired >= available:
            logger.debug(
                "Group %s has %d rows, selecting all available rows because desired=%d",
                group_values,
                available,
                desired,
            )
            sampled_frames.append(group_df)
            actual_sampled += available
        else:
            sampled = group_df.sample(n=desired, random_state=random_state)
            sampled_frames.append(sampled)
            actual_sampled += desired
            logger.debug(
                "Sampled %d rows from group %s out of %d available",
                desired,
                group_values,
                available,
            )

    sampled_df = pd.concat(sampled_frames, ignore_index=True)
    logger.info("Stratified sampling produced %d rows (target was approx %d)", len(sampled_df), target_size)
    return sampled_df


def create_stratified_subset(
    df: pd.DataFrame,
    timestamp_column: str = "author_date",
    start_year: int = 2018,
    end_year: int = 2026,
    group_columns: Sequence[str] = ("project", "buggy"),
    target_size: int = 10000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Filter then sample commits for the training dataset subset.

    This wrapper preserves original timestamps and returns an approximate
    subset of commits from the specified year range.
    """
    filtered = filter_commits_by_date(
        df,
        timestamp_column=timestamp_column,
        start_year=start_year,
        end_year=end_year,
    )
    sampled = stratified_sample(
        filtered,
        group_columns=group_columns,
        target_size=target_size,
        random_state=random_state,
    )
    return sampled


__all__ = ["filter_commits_by_date", "stratified_sample", "create_stratified_subset", "SamplingError"]
