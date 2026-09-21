from __future__ import annotations

import ast
import hashlib
import json
import logging
import zipfile
from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Canonical dataset:
# 59,996 commits + JIT + 768-D CodeBERT + canonical SZZ label
MASTER_DATASET = (
    PROJECT_ROOT
    / "data"
    / "final_multilanguage_59996_szz_buggy.csv"
)

# Raw/structured LLM feature files
LLM_DIR = (
    PROJECT_ROOT
    / "data"
    / "llm_experiment_dataset"
)

LLM_FILES = [
    "train.csv",
    "validation.csv",
    "test.csv",
]

# Output directory
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "public_release"
)

FINAL_DATASET = (
    OUTPUT_DIR
    / "final_multilingual_jit_codebert_llm_59996_768d.csv"
)

FINAL_ZIP = (
    OUTPUT_DIR
    / "final_multilingual_jit_codebert_llm_59996_768d.zip"
)

CHECKSUM_FILE = (
    OUTPUT_DIR
    / "SHA256SUMS.txt"
)

EXPECTED_ROWS = 59_996
EXPECTED_EMBEDDING_DIM = 768


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# Label normalization
# ============================================================

def normalize_buggy_label(value: object) -> int:
    """
    Normalize the buggy label to binary integer form.

    Supported representations:
        0
        1
        False
        True
        "0"
        "1"
        "False"
        "True"

    Returns:
        0 = non-buggy
        1 = buggy
    """

    if pd.isna(value):
        raise ValueError(
            "Missing value found in buggy label."
        )

    # Boolean
    if isinstance(value, bool):
        return int(value)

    # Integer / float
    if isinstance(value, (int, float)):
        if value == 0:
            return 0

        if value == 1:
            return 1

    # String
    text = str(value).strip().lower()

    if text in {"0", "false"}:
        return 0

    if text in {"1", "true"}:
        return 1

    raise ValueError(
        f"Invalid buggy label encountered: {value!r}"
    )


# ============================================================
# Embedding parsing
# ============================================================

def parse_embedding(value: object) -> list[float]:
    """
    Parse a CodeBERT embedding stored as JSON/list text.
    """

    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]

    if pd.isna(value):
        raise ValueError(
            "Missing CodeBERT embedding."
        )

    text = str(value).strip()

    try:
        values = json.loads(text)
    except json.JSONDecodeError:
        values = ast.literal_eval(text)

    if not isinstance(values, (list, tuple)):
        raise ValueError(
            "CodeBERT embedding is not list-like."
        )

    return [float(x) for x in values]


# ============================================================
# Validation helpers
# ============================================================

def validate_unique_commit_ids(
    df: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Verify commit_id exists and is unique."""

    if "commit_id" not in df.columns:
        raise ValueError(
            f"{dataset_name} does not contain commit_id."
        )

    duplicates = df["commit_id"].duplicated().sum()

    if duplicates:
        raise ValueError(
            f"{dataset_name} contains "
            f"{duplicates:,} duplicate commit IDs."
        )


def get_language(project: str) -> str:
    """Infer language from project naming convention."""

    project = str(project)

    if project.startswith("apache/"):
        return "Java"

    if project.startswith("cpp/"):
        return "C++"

    if project.startswith("python/"):
        return "Python"

    return "Unknown"


def sha256_file(path: Path) -> str:
    """Calculate SHA-256 checksum for a file."""

    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# Main
# ============================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # STEP 1
    # Load canonical dataset
    # ========================================================

    logger.info(
        "Loading canonical dataset:\n%s",
        MASTER_DATASET,
    )

    master = pd.read_csv(
        MASTER_DATASET,
        low_memory=False,
    )

    logger.info(
        "Canonical dataset shape: %s",
        master.shape,
    )

    if len(master) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} canonical rows, "
            f"found {len(master):,}."
        )

    validate_unique_commit_ids(
        master,
        "Canonical dataset",
    )

    required_columns = [
        "commit_id",
        "project",
        "buggy",
        "embedding",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in master.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required canonical columns: "
            f"{missing_columns}"
        )

    # ========================================================
    # STEP 2
    # Normalize canonical buggy labels
    # ========================================================

    logger.info(
        "Normalizing canonical buggy labels..."
    )

    master["buggy"] = master["buggy"].apply(
        normalize_buggy_label
    )

    # Verify only 0/1 remain.
    invalid = ~master["buggy"].isin([0, 1])

    if invalid.any():
        raise ValueError(
            "Canonical dataset still contains "
            "invalid buggy labels."
        )

    logger.info(
        "Canonical label distribution:\n%s",
        master["buggy"]
        .value_counts()
        .sort_index()
        .to_string(),
    )

    # ========================================================
    # STEP 3
    # Verify CodeBERT dimensionality
    # ========================================================

    logger.info(
        "Checking CodeBERT embedding dimensionality..."
    )

    embedding_lengths = (
        master["embedding"]
        .map(
            lambda value: len(
                parse_embedding(value)
            )
        )
    )

    invalid_embeddings = (
        embedding_lengths
        != EXPECTED_EMBEDDING_DIM
    ).sum()

    if invalid_embeddings:
        raise ValueError(
            f"{invalid_embeddings:,} embeddings are not "
            f"{EXPECTED_EMBEDDING_DIM}-dimensional."
        )

    logger.info(
        "All %s embeddings are %d-dimensional.",
        f"{len(master):,}",
        EXPECTED_EMBEDDING_DIM,
    )

    # ========================================================
    # STEP 4
    # Load all LLM split files
    # ========================================================

    llm_frames: list[pd.DataFrame] = []

    for filename in LLM_FILES:

        path = LLM_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"LLM file not found:\n{path}"
            )

        logger.info(
            "Loading LLM file:\n%s",
            path,
        )

        llm_split = pd.read_csv(
            path,
            low_memory=False,
        )

        logger.info(
            "%s shape: %s",
            filename,
            llm_split.shape,
        )

        validate_unique_commit_ids(
            llm_split,
            f"LLM {filename}",
        )

        llm_split["commit_id"] = (
            llm_split["commit_id"]
            .astype(str)
            .str.strip()
        )

        # Normalize the LLM buggy label too.
        if "buggy" not in llm_split.columns:
            raise ValueError(
                f"{filename} does not contain buggy."
            )

        llm_split["buggy"] = (
            llm_split["buggy"]
            .apply(normalize_buggy_label)
        )

        llm_frames.append(llm_split)

    # ========================================================
    # STEP 5
    # Combine LLM splits
    # ========================================================

    llm = pd.concat(
        llm_frames,
        ignore_index=True,
    )

    logger.info(
        "Combined LLM dataset shape: %s",
        llm.shape,
    )

    if len(llm) != EXPECTED_ROWS:
        raise ValueError(
            "Combined LLM dataset does not contain "
            f"{EXPECTED_ROWS:,} rows."
        )

    validate_unique_commit_ids(
        llm,
        "Combined LLM dataset",
    )

    # ========================================================
    # STEP 6
    # Normalize canonical commit IDs
    # ========================================================

    master["commit_id"] = (
        master["commit_id"]
        .astype(str)
        .str.strip()
    )

    # ========================================================
    # STEP 7
    # Verify exact commit-ID coverage
    # ========================================================

    logger.info(
        "Checking commit-ID coverage..."
    )

    master_ids = set(
        master["commit_id"]
    )

    llm_ids = set(
        llm["commit_id"]
    )

    missing_from_llm = (
        master_ids - llm_ids
    )

    extra_in_llm = (
        llm_ids - master_ids
    )

    if missing_from_llm:
        raise ValueError(
            f"{len(missing_from_llm):,} canonical commits "
            "are missing from the LLM dataset."
        )

    if extra_in_llm:
        raise ValueError(
            f"{len(extra_in_llm):,} LLM commits "
            "are not present in the canonical dataset."
        )

    logger.info(
        "Commit-ID coverage verified: "
        "59,996 / 59,996."
    )

    # ========================================================
    # STEP 8
    # Verify normalized buggy labels
    # ========================================================

    logger.info(
        "Comparing normalized buggy labels..."
    )

    label_check = master[
        ["commit_id", "buggy"]
    ].merge(
        llm[
            ["commit_id", "buggy"]
        ],
        on="commit_id",
        how="inner",
        validate="one_to_one",
        suffixes=(
            "_canonical",
            "_llm",
        ),
    )

    mismatches = (
        label_check["buggy_canonical"]
        != label_check["buggy_llm"]
    )

    mismatch_count = int(
        mismatches.sum()
    )

    logger.info(
        "Actual buggy-label mismatches: %s",
        f"{mismatch_count:,}",
    )

    if mismatch_count:
        logger.error(
            "Actual label mismatches detected."
        )

        print(
            label_check[mismatches]
            .head(20)
            .to_string(index=False)
        )

        raise ValueError(
            f"Found {mismatch_count:,} actual "
            "buggy-label mismatches."
        )

    logger.info(
        "Buggy labels verified successfully: "
        "0 mismatches."
    )

    # ========================================================
    # STEP 9
    # Remove duplicate buggy column from LLM dataset
    # ========================================================

    llm = llm.drop(
        columns=["buggy"]
    )

    # ========================================================
    # STEP 10
    # Merge LLM features with canonical dataset
    # ========================================================

    logger.info(
        "Merging LLM features using commit_id..."
    )

    final_df = master.merge(
        llm,
        on="commit_id",
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_llm",
        ),
    )

    # ========================================================
    # STEP 11
    # Validate final dataset
    # ========================================================

    logger.info(
        "Validating final merged dataset..."
    )

    if len(final_df) != EXPECTED_ROWS:
        raise ValueError(
            "Final merge changed row count."
        )

    validate_unique_commit_ids(
        final_df,
        "Final dataset",
    )

    # Determine which columns came from LLM.
    master_columns = set(
        master.columns
    )

    llm_feature_columns = [
        column
        for column in final_df.columns
        if column not in master_columns
        and column != "commit_id"
    ]

    if not llm_feature_columns:
        raise ValueError(
            "No LLM features were added."
        )

    logger.info(
        "LLM feature columns added: %d",
        len(llm_feature_columns),
    )

    # Verify no complete row is missing all LLM features.
    missing_llm_rows = (
        final_df[
            llm_feature_columns
        ]
        .isna()
        .all(axis=1)
        .sum()
    )

    if missing_llm_rows:
        raise ValueError(
            f"{missing_llm_rows:,} rows have no LLM features."
        )

    # ========================================================
    # STEP 12
    # Final label distribution
    # ========================================================

    logger.info("=" * 70)
    logger.info("FINAL DATASET")
    logger.info("=" * 70)

    logger.info(
        "Rows       : %s",
        f"{len(final_df):,}",
    )

    logger.info(
        "Columns    : %s",
        len(final_df.columns),
    )

    logger.info(
        "Unique IDs : %s",
        f"{final_df['commit_id'].nunique():,}",
    )

    logger.info(
        "Buggy distribution:\n%s",
        final_df["buggy"]
        .value_counts()
        .sort_index()
        .to_string(),
    )

    # ========================================================
    # STEP 13
    # Project/language statistics
    # ========================================================

    project_info = (
        final_df[
            ["project"]
        ]
        .drop_duplicates()
        .copy()
    )

    project_info["language"] = (
        project_info["project"]
        .apply(get_language)
    )

    logger.info(
        "Distinct projects: %d",
        len(project_info),
    )

    logger.info(
        "Projects by language:\n%s",
        project_info[
            "language"
        ]
        .value_counts()
        .to_string(),
    )

    # ========================================================
    # STEP 14
    # Save final CSV
    # ========================================================

    logger.info(
        "Saving final reproducibility dataset..."
    )

    final_df.to_csv(
        FINAL_DATASET,
        index=False,
    )

    logger.info(
        "Saved:\n%s",
        FINAL_DATASET,
    )

    # ========================================================
    # STEP 15
    # Create ZIP
    # ========================================================

    logger.info(
        "Creating ZIP archive..."
    )

    with zipfile.ZipFile(
        FINAL_ZIP,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:

        archive.write(
            FINAL_DATASET,
            arcname=FINAL_DATASET.name,
        )

    logger.info(
        "Saved ZIP:\n%s",
        FINAL_ZIP,
    )

    # ========================================================
    # STEP 16
    # SHA-256 checksum
    # ========================================================

    logger.info(
        "Calculating SHA-256..."
    )

    checksum = sha256_file(
        FINAL_ZIP
    )

    CHECKSUM_FILE.write_text(
        f"{checksum}  {FINAL_ZIP.name}\n",
        encoding="utf-8",
    )

    logger.info(
        "SHA-256: %s",
        checksum,
    )

    # ========================================================
    # DONE
    # ========================================================

    logger.info("=" * 70)
    logger.info(
        "PUBLIC REPRODUCIBILITY DATASET CREATED"
    )
    logger.info("=" * 70)

    logger.info(
        "CSV : %s",
        FINAL_DATASET,
    )

    logger.info(
        "ZIP : %s",
        FINAL_ZIP,
    )

    logger.info(
        "SHA : %s",
        CHECKSUM_FILE,
    )


if __name__ == "__main__":
    main()