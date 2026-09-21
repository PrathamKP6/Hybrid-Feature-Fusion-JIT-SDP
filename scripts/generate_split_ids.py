"""
generate_split_ids.py

Generate publication-ready train/validation/test commit-ID files.

Source:
    results/chronological_splits/
        train.csv
        validation.csv
        test.csv

Output:
    data/public_release/split_ids/
        train_ids.txt
        validation_ids.txt
        test_ids.txt
        split_summary.txt
"""

from pathlib import Path
import logging
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = ROOT / "results" / "chronological_splits"
OUTPUT_DIR = ROOT / "data" / "public_release" / "split_ids"

SPLITS = {
    "train": SPLIT_DIR / "train.csv",
    "validation": SPLIT_DIR / "validation.csv",
    "test": SPLIT_DIR / "test.csv",
}

EXPECTED_COUNTS = {
    "train": 41_998,
    "validation": 8_998,
    "test": 9_000,
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def main():

    logger.info("=" * 70)
    logger.info("GENERATING PUBLICATION SPLIT IDS")
    logger.info("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    split_ids = {}

    for split_name, path in SPLITS.items():

        if not path.exists():
            raise FileNotFoundError(
                f"{split_name} split not found:\n{path}"
            )

        logger.info("Loading %s split...", split_name)

        df = pd.read_csv(
            path,
            usecols=["commit_id"],
            low_memory=False,
        )

        expected = EXPECTED_COUNTS[split_name]

        if len(df) != expected:
            raise AssertionError(
                f"{split_name}: expected {expected:,} rows, "
                f"found {len(df):,}"
            )

        if df["commit_id"].isna().any():
            raise AssertionError(
                f"{split_name}: missing commit IDs found."
            )

        if df["commit_id"].duplicated().any():
            raise AssertionError(
                f"{split_name}: duplicate commit IDs found."
            )

        ids = df["commit_id"].astype(str).tolist()

        split_ids[split_name] = set(ids)

        output_file = OUTPUT_DIR / f"{split_name}_ids.txt"

        with output_file.open("w", encoding="utf-8") as f:
            for commit_id in ids:
                f.write(commit_id + "\n")

        logger.info(
            "Saved %s: %s IDs",
            output_file,
            f"{len(ids):,}",
        )

    # ------------------------------------------------------------
    # Check overlaps
    # ------------------------------------------------------------

    train_ids = split_ids["train"]
    validation_ids = split_ids["validation"]
    test_ids = split_ids["test"]

    if train_ids & validation_ids:
        raise AssertionError(
            "Train/validation overlap detected."
        )

    if train_ids & test_ids:
        raise AssertionError(
            "Train/test overlap detected."
        )

    if validation_ids & test_ids:
        raise AssertionError(
            "Validation/test overlap detected."
        )

    logger.info("PASS: no split overlap.")

    # ------------------------------------------------------------
    # Check complete coverage
    # ------------------------------------------------------------

    combined = train_ids | validation_ids | test_ids

    expected_total = sum(EXPECTED_COUNTS.values())

    if len(combined) != expected_total:
        raise AssertionError(
            f"Expected {expected_total:,} unique IDs, "
            f"found {len(combined):,}"
        )

    logger.info(
        "PASS: complete coverage = %s IDs.",
        f"{len(combined):,}",
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    summary_file = OUTPUT_DIR / "split_summary.txt"

    summary = [
        "PUBLICATION DATASET SPLIT SUMMARY",
        "=" * 70,
        "",
        "Split policy:",
        "Project-wise chronological 70/15/15 split.",
        "Each project is ordered chronologically and divided into:",
        "  - oldest 70%  -> training",
        "  - next 15%    -> validation",
        "  - newest 15%  -> testing",
        "",
        "No random shuffling.",
        "No random train/test split.",
        "No resampling.",
        "",
        "SPLIT SIZES",
        "-" * 70,
        f"Train:      {len(train_ids):,}",
        f"Validation: {len(validation_ids):,}",
        f"Test:       {len(test_ids):,}",
        f"Total:      {len(combined):,}",
        "",
        "OVERLAP CHECK",
        "-" * 70,
        "Train ∩ Validation = 0",
        "Train ∩ Test       = 0",
        "Validation ∩ Test  = 0",
        "",
        "Coverage: PASS",
        "",
    ]

    summary_file.write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    logger.info("Saved summary:")
    logger.info("%s", summary_file)

    logger.info("=" * 70)
    logger.info("SPLIT ID GENERATION PASSED")
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        logger.error("SPLIT ID GENERATION FAILED")
        logger.error("%s", error)
        sys.exit(1)