"""Data loading utilities for the Hybrid Feature Fusion project.

Provides a single public function ``load_commits_csv`` that loads a CSV into
a pandas DataFrame and validates the presence of required columns. The loader
parses commit timestamps robustly: it accepts string datetimes and numeric
epoch timestamps (seconds or milliseconds). The function raises informative
exceptions on failure.

Design notes:
- Default required columns are adapted to the provided dataset:
  ``commit_id``, ``author_date``, ``buggy``, ``project``.
- The loader does not perform cleaning or sampling.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import logging
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Base exception for data loading errors."""


class MissingColumnsError(DataLoadError):
    """Raised when required columns are missing from the loaded DataFrame.

    Attributes:
        missing: list[str] -- list of missing column names
    """

    def __init__(self, missing: Sequence[str], message: Optional[str] = None) -> None:
        self.missing: List[str] = list(missing)
        if message is None:
            message = f"Missing required columns: {', '.join(self.missing)}"
        super().__init__(message)


def _infer_epoch_unit(max_val: int) -> str:
    """Infer epoch unit from maximum numeric magnitude.

    Returns one of: 's' (seconds), 'ms' (milliseconds), or 'ns' (nanoseconds).
    """
    if max_val > 10 ** 15:
        return "ns"
    if max_val > 10 ** 12:
        return "ms"
    return "s"


def load_commits_csv(
    file_path: str | Path,
    required_columns: Optional[Iterable[str]] = None,
    parse_dates: bool = True,
    date_columns: Sequence[str] = ("author_date",),
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load commit data from a CSV into a pandas DataFrame.

    Args:
        file_path: Path to the CSV file to load.
        required_columns: Iterable of columns that must be present. If
            ``None``, defaults to ("commit_id", "author_date", "buggy", "project").
        parse_dates: Whether to attempt parsing columns in ``date_columns``.
        date_columns: Sequence of column names to parse as datetimes.
        nrows: If set, only read that many rows (useful for quick checks).

    Returns:
        A pandas DataFrame with parsed date columns when possible.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        DataLoadError: For read/parse errors.
        MissingColumnsError: If required columns are absent.
    """
    p = Path(file_path)
    if not p.exists():
        logger.error("CSV file not found: %s", p)
        raise FileNotFoundError(f"CSV file not found: {p}")

    default_required = ("commit_id", "author_date", "buggy", "project")
    req_cols = list(required_columns) if required_columns is not None else list(default_required)

    try:
        df = pd.read_csv(p, nrows=nrows)
    except Exception as exc:
        logger.exception("Failed to read CSV '%s': %s", p, exc)
        raise DataLoadError(f"Failed to read CSV '{p}': {exc}") from exc

    # Validate required columns
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        logger.error("CSV is missing required columns: %s", missing)
        raise MissingColumnsError(missing)

    # Parse date-like columns robustly
    if parse_dates:
        for col in date_columns:
            if col not in df.columns:
                continue

            # Skip if already datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                continue

            # Try to interpret column as numeric epoch first
            numeric = pd.to_numeric(df[col], errors="coerce")
            num_non_na = int(numeric.notna().sum())
            try:
                if num_non_na >= max(1, int(len(df) * 0.1)):
                    # If many values parse as numeric, treat as epoch
                    max_val = int(numeric.abs().max(skipna=True))
                    unit = _infer_epoch_unit(max_val)
                    df[col] = pd.to_datetime(numeric, unit=unit, errors="raise")
                    logger.info("Parsed numeric date column '%s' as epoch (%s)", col, unit)
                    continue
            except Exception as exc:
                logger.debug("Numeric epoch parse failed for column '%s': %s", col, exc)

            # Fallback: parse strings
            try:
                df[col] = pd.to_datetime(df[col], errors="raise", infer_datetime_format=True)
                logger.info("Parsed string date column '%s' using to_datetime", col)
            except Exception as exc:
                # As a last resort, coerce with errors='coerce' and warn
                logger.warning(
                    "Failed to strictly parse date column '%s' (%s). Falling back to coercion.",
                    col,
                    exc,
                )
                df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)

    logger.info("Loaded CSV '%s' with %d rows and %d columns", p, len(df), len(df.columns))
    return df


__all__ = ["load_commits_csv", "DataLoadError", "MissingColumnsError"]
