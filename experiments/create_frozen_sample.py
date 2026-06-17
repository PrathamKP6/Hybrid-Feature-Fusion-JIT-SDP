"""Generate a frozen, reusable sample dataset for all experiments.

This script creates a consistent, reproducible subset of the full dataset
by filtering commits from 2018-2026 and applying stratified sampling
(stratified by project and buggy label). The frozen sample is saved for
use across all experiments to ensure reproducibility.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.load_data import load_commits_csv
from preprocessing.sampling import filter_commits_by_date, stratified_sample

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = PROJECT_ROOT / "final_multilanguage_dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FROZEN_SAMPLE_PATH = RESULTS_DIR / "frozen_sample_dataset.csv"
FROZEN_COMMIT_IDS_PATH = RESULTS_DIR / "frozen_sample_commit_ids.csv"


def main() -> None:
    """Generate and save frozen sample dataset."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset from %s", DATA_PATH)
    df = load_commits_csv(DATA_PATH)
    logger.info("Loaded %d rows from dataset", len(df))

    logger.info("Filtering commits by date range 2018-2026")
    df_filtered = filter_commits_by_date(
        df,
        timestamp_column="author_date",
        start_year=2018,
        end_year=2026,
    )
    logger.info("After date filtering: %d rows", len(df_filtered))

    logger.info("Applying stratified sampling (project × buggy, target=10000, random_state=42)")
    df_sampled = stratified_sample(
        df_filtered,
        group_columns=("project", "buggy"),
        target_size=10000,
        random_state=42,
    )
    logger.info("After stratified sampling: %d rows", len(df_sampled))

    logger.info("Saving frozen sample to %s", FROZEN_SAMPLE_PATH)
    df_sampled.to_csv(FROZEN_SAMPLE_PATH, index=False)
    logger.info("Saved frozen sample dataset")

    logger.info("Saving commit IDs to %s", FROZEN_COMMIT_IDS_PATH)
    commit_ids = df_sampled[["commit_id"]].copy()
    commit_ids.to_csv(FROZEN_COMMIT_IDS_PATH, index=False)
    logger.info("Saved %d commit IDs", len(commit_ids))

    logger.info("Frozen sample creation complete!")
    logger.info("Dataset size: %d rows, %d columns", len(df_sampled), len(df_sampled.columns))
    logger.info("Class distribution (buggy): %s", df_sampled["buggy"].value_counts().to_dict())
    logger.info("Project distribution: %s", df_sampled["project"].value_counts().head(10).to_dict())


if __name__ == "__main__":
    main()
