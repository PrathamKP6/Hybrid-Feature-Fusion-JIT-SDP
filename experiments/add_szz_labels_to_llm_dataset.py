import shutil
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Raw LLM dataset
LLM_DIR = (
    PROJECT_ROOT
    / "data"
    / "llm_experiment_dataset"
)

# Canonical dataset containing commit_id + buggy
SZZ_DATASET = (
    PROJECT_ROOT
    / "data"
    / "final_multilanguage_59996_szz_buggy.csv"
)

# Backup directory
BACKUP_DIR = (
    LLM_DIR
    / "backup_before_szz_label_merge"
)


# ============================================================
# SPLITS
# ============================================================

SPLITS = [
    "train",
    "validation",
    "test",
]


# ============================================================
# EXPECTED ROW COUNTS
# ============================================================

EXPECTED_ROWS = {
    "train": 41998,
    "validation": 8998,
    "test": 9000,
}


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("ADD SZZ BUGGY LABELS TO RAW LLM DATASET")
    print("=" * 70)

    print(
        f"\nLLM dataset directory:\n"
        f"  {LLM_DIR}"
    )

    print(
        f"\nCanonical SZZ dataset:\n"
        f"  {SZZ_DATASET}"
    )

    # --------------------------------------------------------
    # Verify paths
    # --------------------------------------------------------

    if not LLM_DIR.exists():
        raise FileNotFoundError(
            f"LLM dataset directory not found:\n{LLM_DIR}"
        )

    if not SZZ_DATASET.exists():
        raise FileNotFoundError(
            f"Canonical SZZ dataset not found:\n{SZZ_DATASET}"
        )

    # --------------------------------------------------------
    # Load canonical SZZ dataset
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING CANONICAL SZZ DATASET")
    print("=" * 70)

    print(f"Path: {SZZ_DATASET}")

    szz_df = pd.read_csv(
        SZZ_DATASET,
        low_memory=False,
        usecols=["commit_id", "buggy"],
    )

    print(
        f"Rows: {len(szz_df):,}"
    )

    print(
        f"Columns: {list(szz_df.columns)}"
    )

    # --------------------------------------------------------
    # Verify canonical dataset
    # --------------------------------------------------------

    print("\nVerifying canonical SZZ dataset...")

    if szz_df["commit_id"].isna().any():
        raise ValueError(
            "Canonical SZZ dataset contains "
            "missing commit_id values."
        )

    if szz_df["buggy"].isna().any():
        raise ValueError(
            "Canonical SZZ dataset contains "
            "missing buggy labels."
        )

    duplicate_ids = int(
        szz_df["commit_id"].duplicated().sum()
    )

    print(
        f"Duplicate commit IDs: {duplicate_ids}"
    )

    if duplicate_ids > 0:
        raise ValueError(
            "Canonical SZZ dataset contains duplicate "
            "commit IDs. Cannot safely perform a "
            "one-to-one label merge."
        )

    # Normalize buggy labels
    def normalize_buggy(value):

        if isinstance(value, bool):
            return int(value)

        value_str = str(value).strip().lower()

        if value_str in {"0", "false"}:
            return 0

        if value_str in {"1", "true"}:
            return 1

        raise ValueError(
            f"Unexpected buggy value in canonical dataset: "
            f"{value!r}"
        )

    szz_df["buggy"] = (
        szz_df["buggy"]
        .apply(normalize_buggy)
        .astype(int)
    )

    # --------------------------------------------------------
    # Create lookup table
    # --------------------------------------------------------

    label_lookup = (
        szz_df
        .set_index("commit_id")["buggy"]
    )

    print(
        "\n[OK] Canonical commit_id → buggy lookup created."
    )

    # --------------------------------------------------------
    # Create backup directory
    # --------------------------------------------------------

    BACKUP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\nBackups will be stored in:\n"
        f"  {BACKUP_DIR}"
    )

    # --------------------------------------------------------
    # Process each split
    # --------------------------------------------------------

    for split in SPLITS:

        print("\n" + "=" * 70)
        print(
            f"PROCESSING {split.upper()} SPLIT"
        )
        print("=" * 70)

        input_path = (
            LLM_DIR
            / f"{split}.csv"
        )

        backup_path = (
            BACKUP_DIR
            / f"{split}.csv"
        )

        if not input_path.exists():
            raise FileNotFoundError(
                f"Missing LLM split:\n{input_path}"
            )

        # ----------------------------------------------------
        # Load raw LLM dataset
        # ----------------------------------------------------

        print(
            f"Loading:\n  {input_path}"
        )

        llm_df = pd.read_csv(
            input_path,
            low_memory=False,
        )

        print(
            f"Rows:    {len(llm_df):,}"
        )

        print(
            f"Columns: {len(llm_df.columns)}"
        )

        # ----------------------------------------------------
        # Verify expected row count
        # ----------------------------------------------------

        expected_rows = EXPECTED_ROWS[split]

        if len(llm_df) != expected_rows:
            raise ValueError(
                f"{split} row count mismatch.\n"
                f"Expected: {expected_rows:,}\n"
                f"Found:    {len(llm_df):,}"
            )

        # ----------------------------------------------------
        # Verify commit_id
        # ----------------------------------------------------

        if "commit_id" not in llm_df.columns:
            raise ValueError(
                f"{split}.csv does not contain commit_id."
            )

        if llm_df["commit_id"].isna().any():
            raise ValueError(
                f"{split}.csv contains missing commit_id."
            )

        duplicate_ids = int(
            llm_df["commit_id"].duplicated().sum()
        )

        print(
            f"Duplicate LLM commit IDs: "
            f"{duplicate_ids}"
        )

        if duplicate_ids > 0:
            raise ValueError(
                f"{split}.csv contains duplicate "
                f"commit IDs."
            )

        # ----------------------------------------------------
        # Check whether buggy already exists
        # ----------------------------------------------------

        if "buggy" in llm_df.columns:

            raise ValueError(
                f"{split}.csv already contains a "
                f"'buggy' column.\n"
                f"Refusing to overwrite it."
            )

        # ----------------------------------------------------
        # Match commit IDs
        # ----------------------------------------------------

        print(
            "\nMatching LLM commit IDs "
            "against canonical SZZ dataset..."
        )

        matched_labels = (
            llm_df["commit_id"]
            .map(label_lookup)
        )

        matched_count = int(
            matched_labels.notna().sum()
        )

        unmatched_count = int(
            matched_labels.isna().sum()
        )

        print(
            f"Matched:   {matched_count:,}"
        )

        print(
            f"Unmatched: {unmatched_count:,}"
        )

        # ----------------------------------------------------
        # CRITICAL: every commit must match
        # ----------------------------------------------------

        if unmatched_count > 0:

            unmatched_ids = (
                llm_df.loc[
                    matched_labels.isna(),
                    "commit_id",
                ]
                .head(20)
                .tolist()
            )

            raise ValueError(
                f"\n{unmatched_count:,} commit IDs in "
                f"{split}.csv were not found in the "
                f"canonical SZZ dataset.\n\n"
                f"First unmatched IDs:\n"
                f"{unmatched_ids}\n\n"
                f"No files were modified for this split."
            )

        # ----------------------------------------------------
        # Add buggy label
        # ----------------------------------------------------

        llm_df["buggy"] = (
            matched_labels
            .astype(int)
        )

        # ----------------------------------------------------
        # Verify resulting labels
        # ----------------------------------------------------

        if llm_df["buggy"].isna().any():
            raise ValueError(
                f"Missing buggy labels remain "
                f"after merge for {split}."
            )

        unique_labels = set(
            llm_df["buggy"].unique()
        )

        if not unique_labels.issubset({0, 1}):
            raise ValueError(
                f"Unexpected buggy labels in "
                f"{split}: {unique_labels}"
            )

        print(
            "\nBuggy distribution:"
        )

        print(
            llm_df["buggy"]
            .value_counts()
            .sort_index()
            .to_string()
        )

        # ----------------------------------------------------
        # Verify row count unchanged
        # ----------------------------------------------------

        if len(llm_df) != expected_rows:
            raise ValueError(
                f"Row count changed unexpectedly "
                f"for {split}."
            )

        # ----------------------------------------------------
        # BACKUP ORIGINAL FILE
        # ----------------------------------------------------

        print(
            f"\nCreating backup:\n"
            f"  {backup_path}"
        )

        shutil.copy2(
            input_path,
            backup_path,
        )

        # ----------------------------------------------------
        # Write modified dataset IN PLACE
        # ----------------------------------------------------

        print(
            f"Writing modified dataset:\n"
            f"  {input_path}"
        )

        llm_df.to_csv(
            input_path,
            index=False,
        )

        print(
            f"[OK] {split}.csv updated."
        )

    # ========================================================
    # FINAL VERIFICATION
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL VERIFICATION")
    print("=" * 70)

    for split in SPLITS:

        path = (
            LLM_DIR
            / f"{split}.csv"
        )

        df = pd.read_csv(
            path,
            low_memory=False,
        )

        print(
            f"\n{split.upper()}:"
        )

        print(
            f"  Rows:    {len(df):,}"
        )

        print(
            f"  Columns: {len(df.columns)}"
        )

        print(
            f"  buggy present: "
            f"{'buggy' in df.columns}"
        )

        print(
            f"  commit_id present: "
            f"{'commit_id' in df.columns}"
        )

        print(
            f"  Missing buggy: "
            f"{df['buggy'].isna().sum()}"
        )

        print(
            f"  Duplicate IDs: "
            f"{df['commit_id'].duplicated().sum()}"
        )

    print("\n" + "=" * 70)
    print("LABEL MERGE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        "\nModified files:"
    )

    for split in SPLITS:
        print(
            f"  {LLM_DIR / f'{split}.csv'}"
        )

    print(
        "\nOriginal files backed up to:"
    )

    print(
        f"  {BACKUP_DIR}"
    )

    print(
        "\nYou can now run:"
    )

    print(
        "  python experiments/experiment_3_llm_only.py"
    )


if __name__ == "__main__":
    main()