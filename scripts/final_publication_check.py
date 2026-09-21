"""
final_publication_check.py

Final consistency check for the complete publication/reproducibility package.

This script does NOT modify the dataset or publication artifacts.

Checks:
    1. Required publication files exist.
    2. Final CSV has the expected 44-column schema.
    3. Final CSV has 59,996 rows and unique commit IDs.
    4. Target labels are 0/1 with expected counts.
    5. Required JIT and LLM columns exist.
    6. CodeBERT embeddings are 768-dimensional.
    7. Split-ID files exist and contain:
         train      = 41,998
         validation = 8,998
         test       = 9,000
    8. Split IDs have no overlap.
    9. Split IDs cover all dataset commit IDs.
    10. ZIP exists and contains exactly the final CSV.
    11. SHA256SUMS.txt contains all expected checksums.
    12. Recomputed SHA-256 values match SHA256SUMS.txt.
    13. No unexpected files exist in data/public_release/.
    14. Documentation files exist.

Output:
    Console report only.

IMPORTANT:
    This script is read-only.
    It does not modify any publication artifact.
"""

from pathlib import Path
import ast
import hashlib
import logging
import sys
import zipfile

import pandas as pd


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

PUBLIC_DIR = ROOT / "data" / "public_release"

DATASET = (
    PUBLIC_DIR
    / "final_multilingual_jit_codebert_llm_59996_768d.csv"
)

ZIP_FILE = (
    PUBLIC_DIR
    / "final_multilingual_jit_codebert_llm_59996_768d.zip"
)

CHECKSUM_FILE = PUBLIC_DIR / "SHA256SUMS.txt"

AUDIT_REPORT = PUBLIC_DIR / "AUDIT_REPORT.txt"
DATASET_MD = PUBLIC_DIR / "DATASET.md"
FEATURE_DICTIONARY_CSV = PUBLIC_DIR / "FEATURE_DICTIONARY.csv"
FEATURE_DICTIONARY_MD = PUBLIC_DIR / "FEATURE_DICTIONARY.md"

SPLIT_DIR = PUBLIC_DIR / "split_ids"

TRAIN_IDS = SPLIT_DIR / "train_ids.txt"
VALIDATION_IDS = SPLIT_DIR / "validation_ids.txt"
TEST_IDS = SPLIT_DIR / "test_ids.txt"
SPLIT_SUMMARY = SPLIT_DIR / "split_summary.txt"


# ----------------------------------------------------------------------
# Expected dataset schema
# ----------------------------------------------------------------------

EXPECTED_COLUMNS = [
    "commit_id",
    "project",
    "buggy",
    "fix",
    "year",
    "author_date",
    "la",
    "ld",
    "nf",
    "nd",
    "ns",
    "ent",
    "ndev",
    "age",
    "nuc",
    "aexp",
    "arexp",
    "asexp",
    "message",
    "diff",
    "embedding",
    "intent",
    "intent_confidence",
    "intent_margin",
    "change",
    "change_confidence",
    "change_margin",
    "risk",
    "risk_confidence",
    "risk_margin",
    "complexity",
    "complexity_confidence",
    "complexity_margin",
    "scope",
    "scope_confidence",
    "scope_margin",
    "test",
    "test_confidence",
    "test_margin",
    "security",
    "security_confidence",
    "security_margin",
    "model",
    "schema_version",
]


# ----------------------------------------------------------------------
# Expected values
# ----------------------------------------------------------------------

EXPECTED_ROWS = 59_996

EXPECTED_LABEL_COUNTS = {
    0: 44_626,
    1: 15_370,
}

EXPECTED_SPLIT_COUNTS = {
    "train": 41_998,
    "validation": 8_998,
    "test": 9_000,
}

EXPECTED_EMBEDDING_DIM = 768


# ----------------------------------------------------------------------
# Feature groups
# ----------------------------------------------------------------------

JIT_FEATURES = [
    "la",
    "ld",
    "nf",
    "ns",
    "nd",
    "ent",
    "ndev",
    "age",
    "nuc",
    "aexp",
    "arexp",
    "asexp",
]

LLM_CATEGORICAL = [
    "intent",
    "change",
    "risk",
    "complexity",
    "scope",
    "test",
    "security",
]

LLM_CONFIDENCE = [
    "intent_confidence",
    "change_confidence",
    "risk_confidence",
    "complexity_confidence",
    "scope_confidence",
    "test_confidence",
    "security_confidence",
]

LLM_MARGIN = [
    "intent_margin",
    "change_margin",
    "risk_margin",
    "complexity_margin",
    "scope_margin",
    "test_margin",
    "security_margin",
]

LLM_PROVENANCE = [
    "model",
    "schema_version",
]


# ----------------------------------------------------------------------
# Publication files that must exist
# ----------------------------------------------------------------------

EXPECTED_PUBLICATION_FILES = [
    DATASET,
    ZIP_FILE,
    CHECKSUM_FILE,
    AUDIT_REPORT,
    DATASET_MD,
    FEATURE_DICTIONARY_CSV,
    FEATURE_DICTIONARY_MD,
    TRAIN_IDS,
    VALIDATION_IDS,
    TEST_IDS,
    SPLIT_SUMMARY,
]


# These are the files that should be represented in SHA256SUMS.txt.
EXPECTED_CHECKSUM_RELATIVE_PATHS = [
    "AUDIT_REPORT.txt",
    "DATASET.md",
    "FEATURE_DICTIONARY.csv",
    "FEATURE_DICTIONARY.md",
    "final_multilingual_jit_codebert_llm_59996_768d.csv",
    "final_multilingual_jit_codebert_llm_59996_768d.zip",
    "split_ids/split_summary.txt",
    "split_ids/test_ids.txt",
    "split_ids/train_ids.txt",
    "split_ids/validation_ids.txt",
]


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def parse_embedding(value):
    """
    Parse a serialized embedding.

    Returns:
        list or None
    """

    if pd.isna(value):
        return None

    if isinstance(value, (list, tuple)):
        return list(value)

    try:
        parsed = ast.literal_eval(str(value).strip())

        if isinstance(parsed, (list, tuple)):
            return list(parsed)

    except Exception:
        return None

    return None


def calculate_sha256(path, chunk_size=1024 * 1024):
    """
    Calculate SHA-256 using chunked binary reads.
    """

    sha256 = hashlib.sha256()

    with path.open("rb") as file:

        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def load_split_ids(path, split_name):
    """
    Load and validate a split-ID file.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"{split_name} split-ID file missing:\n{path}"
        )

    with path.open("r", encoding="utf-8") as file:

        ids = [
            line.strip()
            for line in file
            if line.strip()
        ]

    if len(ids) != EXPECTED_SPLIT_COUNTS[split_name]:

        raise AssertionError(
            f"{split_name}: expected "
            f"{EXPECTED_SPLIT_COUNTS[split_name]:,} IDs, "
            f"found {len(ids):,}"
        )

    if len(ids) != len(set(ids)):

        raise AssertionError(
            f"{split_name}: duplicate commit IDs found."
        )

    return set(ids)


# ----------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------

def check_required_files():
    """
    Check that all expected publication artifacts exist.
    """

    logger.info("-" * 70)
    logger.info("CHECK 1: Required publication files")
    logger.info("-" * 70)

    missing = []

    for path in EXPECTED_PUBLICATION_FILES:

        if not path.exists():
            missing.append(path)

        else:
            logger.info(
                "PASS: %s",
                path.relative_to(PUBLIC_DIR),
            )

    if missing:

        raise AssertionError(
            "Missing publication files:\n"
            + "\n".join(str(path) for path in missing)
        )

    logger.info(
        "PASS: all %d required publication files exist.",
        len(EXPECTED_PUBLICATION_FILES),
    )


def check_dataset():
    """
    Validate the final publication dataset.
    """

    logger.info("-" * 70)
    logger.info("CHECK 2: Final publication dataset")
    logger.info("-" * 70)

    # Read only the header first.
    header = pd.read_csv(
        DATASET,
        nrows=0,
        low_memory=False,
    )

    actual_columns = list(header.columns)

    if actual_columns != EXPECTED_COLUMNS:

        missing = [
            column
            for column in EXPECTED_COLUMNS
            if column not in actual_columns
        ]

        extra = [
            column
            for column in actual_columns
            if column not in EXPECTED_COLUMNS
        ]

        message = (
            "Publication dataset schema mismatch."
        )

        if missing:
            message += (
                f"\nMissing columns: {missing}"
            )

        if extra:
            message += (
                f"\nUnexpected columns: {extra}"
            )

        raise AssertionError(message)

    logger.info(
        "PASS: exact 44-column schema."
    )

    logger.info(
        "Loading dataset for final integrity checks..."
    )

    df = pd.read_csv(
        DATASET,
        low_memory=False,
    )

    if len(df) != EXPECTED_ROWS:

        raise AssertionError(
            f"Expected {EXPECTED_ROWS:,} rows, "
            f"found {len(df):,}"
        )

    logger.info(
        "PASS: row count = %s",
        f"{len(df):,}",
    )

    if df["commit_id"].isna().any():

        raise AssertionError(
            "Missing commit IDs found."
        )

    if df["commit_id"].duplicated().any():

        raise AssertionError(
            "Duplicate commit IDs found."
        )

    logger.info(
        "PASS: %s unique commit IDs.",
        f"{df['commit_id'].nunique():,}",
    )

    # Normalize labels for checking.
    def normalize_label(value):

        if isinstance(value, bool):
            return int(value)

        if isinstance(value, (int, float)):

            if value in (0, 1):
                return int(value)

        text = str(value).strip().lower()

        if text in {"0", "false"}:
            return 0

        if text in {"1", "true"}:
            return 1

        raise ValueError(
            f"Invalid buggy label: {value!r}"
        )

    labels = df["buggy"].apply(normalize_label)

    actual_counts = labels.value_counts().to_dict()

    actual_counts = {
        int(key): int(value)
        for key, value in actual_counts.items()
    }

    if actual_counts != EXPECTED_LABEL_COUNTS:

        raise AssertionError(
            "Unexpected label distribution.\n"
            f"Expected: {EXPECTED_LABEL_COUNTS}\n"
            f"Found: {actual_counts}"
        )

    logger.info(
        "PASS: label distribution = 0:%s, 1:%s",
        f"{EXPECTED_LABEL_COUNTS[0]:,}",
        f"{EXPECTED_LABEL_COUNTS[1]:,}",
    )

    # JIT features.
    missing_jit = [
        feature
        for feature in JIT_FEATURES
        if feature not in df.columns
    ]

    if missing_jit:

        raise AssertionError(
            "Missing JIT features: "
            + ", ".join(missing_jit)
        )

    logger.info(
        "PASS: all %d JIT features present.",
        len(JIT_FEATURES),
    )

    # LLM columns.
    expected_llm = (
        LLM_CATEGORICAL
        + LLM_CONFIDENCE
        + LLM_MARGIN
        + LLM_PROVENANCE
    )

    missing_llm = [
        feature
        for feature in expected_llm
        if feature not in df.columns
    ]

    if missing_llm:

        raise AssertionError(
            "Missing LLM/provenance columns: "
            + ", ".join(missing_llm)
        )

    logger.info(
        "PASS: all LLM semantic/provenance columns present."
    )

    # Embeddings.
    logger.info(
        "Checking 768-dimensional CodeBERT embeddings..."
    )

    dimensions = set()
    invalid_embeddings = 0

    for value in df["embedding"]:

        embedding = parse_embedding(value)

        if embedding is None:

            invalid_embeddings += 1

            continue

        dimensions.add(len(embedding))

    if invalid_embeddings:

        raise AssertionError(
            f"Invalid embeddings: "
            f"{invalid_embeddings:,}"
        )

    if dimensions != {EXPECTED_EMBEDDING_DIM}:

        raise AssertionError(
            f"Unexpected embedding dimensions: "
            f"{dimensions}"
        )

    logger.info(
        "PASS: all embeddings are %d-dimensional.",
        EXPECTED_EMBEDDING_DIM,
    )

    # Missing values.
    missing_values = df.isna().sum()

    missing_values = missing_values[
        missing_values > 0
    ]

    if len(missing_values) > 0:

        details = "\n".join(
            f"  {column}: {count:,}"
            for column, count
            in missing_values.items()
        )

        raise AssertionError(
            "Missing values detected:\n"
            + details
        )

    logger.info(
        "PASS: no missing values."
    )

    return set(
        df["commit_id"].astype(str)
    )


def check_splits(dataset_ids):
    """
    Validate exact split sizes, overlap, and coverage.
    """

    logger.info("-" * 70)
    logger.info("CHECK 3: Split-ID consistency")
    logger.info("-" * 70)

    train_ids = load_split_ids(
        TRAIN_IDS,
        "train",
    )

    validation_ids = load_split_ids(
        VALIDATION_IDS,
        "validation",
    )

    test_ids = load_split_ids(
        TEST_IDS,
        "test",
    )

    logger.info(
        "PASS: train = %s",
        f"{len(train_ids):,}",
    )

    logger.info(
        "PASS: validation = %s",
        f"{len(validation_ids):,}",
    )

    logger.info(
        "PASS: test = %s",
        f"{len(test_ids):,}",
    )

    train_validation = train_ids & validation_ids
    train_test = train_ids & test_ids
    validation_test = validation_ids & test_ids

    if train_validation:

        raise AssertionError(
            "Train/validation overlap found."
        )

    if train_test:

        raise AssertionError(
            "Train/test overlap found."
        )

    if validation_test:

        raise AssertionError(
            "Validation/test overlap found."
        )

    logger.info(
        "PASS: no split overlap."
    )

    combined = (
        train_ids
        | validation_ids
        | test_ids
    )

    if combined != dataset_ids:

        missing = dataset_ids - combined
        extra = combined - dataset_ids

        raise AssertionError(
            "Split coverage mismatch.\n"
            f"Dataset IDs missing from splits: "
            f"{len(missing):,}\n"
            f"Split IDs not found in dataset: "
            f"{len(extra):,}"
        )

    logger.info(
        "PASS: split IDs cover all %s dataset commits.",
        f"{len(dataset_ids):,}",
    )

    return {
        "train": train_ids,
        "validation": validation_ids,
        "test": test_ids,
    }


def check_zip():
    """
    Check ZIP integrity and contents.
    """

    logger.info("-" * 70)
    logger.info("CHECK 4: ZIP integrity")
    logger.info("-" * 70)

    if not zipfile.is_zipfile(ZIP_FILE):

        raise AssertionError(
            "Publication ZIP is not a valid ZIP archive."
        )

    with zipfile.ZipFile(
        ZIP_FILE,
        "r",
    ) as archive:

        bad_file = archive.testzip()

        if bad_file is not None:

            raise AssertionError(
                f"Corrupted ZIP member: {bad_file}"
            )

        names = archive.namelist()

    expected = {
        DATASET.name
    }

    actual = {
        Path(name).name
        for name in names
    }

    if actual != expected:

        raise AssertionError(
            "Unexpected ZIP contents.\n"
            f"Expected: {expected}\n"
            f"Found: {actual}"
        )

    logger.info(
        "PASS: ZIP is valid."
    )

    logger.info(
        "PASS: ZIP contains exactly the final CSV."
    )


def parse_checksum_file():
    """
    Parse SHA256SUMS.txt.
    """

    checksums = {}

    with CHECKSUM_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            parts = line.split(None, 1)

            if len(parts) != 2:

                raise AssertionError(
                    "Malformed checksum line "
                    f"{line_number}: {line}"
                )

            digest, relative_path = parts

            if len(digest) != 64:

                raise AssertionError(
                    "Invalid SHA-256 digest on line "
                    f"{line_number}: {digest}"
                )

            relative_path = relative_path.strip()

            if relative_path in checksums:

                raise AssertionError(
                    "Duplicate checksum entry: "
                    f"{relative_path}"
                )

            checksums[relative_path] = digest.lower()

    return checksums


def check_checksums():
    """
    Recalculate and verify every SHA-256 checksum.
    """

    logger.info("-" * 70)
    logger.info("CHECK 5: SHA-256 checksum consistency")
    logger.info("-" * 70)

    checksums = parse_checksum_file()

    expected_paths = set(
        EXPECTED_CHECKSUM_RELATIVE_PATHS
    )

    actual_paths = set(
        checksums.keys()
    )

    missing_entries = (
        expected_paths - actual_paths
    )

    unexpected_entries = (
        actual_paths - expected_paths
    )

    if missing_entries:

        raise AssertionError(
            "Missing checksum entries:\n"
            + "\n".join(
                sorted(missing_entries)
            )
        )

    if unexpected_entries:

        raise AssertionError(
            "Unexpected checksum entries:\n"
            + "\n".join(
                sorted(unexpected_entries)
            )
        )

    logger.info(
        "PASS: checksum file contains exactly %d expected files.",
        len(expected_paths),
    )

    for relative_path in sorted(expected_paths):

        path = PUBLIC_DIR / relative_path

        if not path.exists():

            raise AssertionError(
                f"Checksum references missing file: "
                f"{relative_path}"
            )

        calculated = calculate_sha256(path)

        recorded = checksums[relative_path]

        if calculated.lower() != recorded.lower():

            raise AssertionError(
                f"SHA-256 mismatch: {relative_path}\n"
                f"Recorded:   {recorded}\n"
                f"Calculated: {calculated}"
            )

        logger.info(
            "PASS: %s",
            relative_path,
        )

    logger.info(
        "PASS: all SHA-256 checksums match."
    )


def check_no_unexpected_files():
    """
    Check for unexpected files in the publication directory.

    Expected:
        publication root files listed above
        split_ids directory

    The checksum file itself is excluded from checksum verification
    but is still an expected publication artifact.
    """

    logger.info("-" * 70)
    logger.info("CHECK 6: Publication directory contents")
    logger.info("-" * 70)

    expected_root_files = {
        path.name
        for path in EXPECTED_PUBLICATION_FILES
        if path.parent == PUBLIC_DIR
    }

    expected_directories = {
        SPLIT_DIR.name
    }

    actual_root_files = {
        path.name
        for path in PUBLIC_DIR.iterdir()
        if path.is_file()
    }

    actual_directories = {
        path.name
        for path in PUBLIC_DIR.iterdir()
        if path.is_dir()
    }

    unexpected_files = (
        actual_root_files - expected_root_files
    )

    unexpected_directories = (
        actual_directories - expected_directories
    )

    if unexpected_files:

        raise AssertionError(
            "Unexpected files in publication directory:\n"
            + "\n".join(
                sorted(unexpected_files)
            )
        )

    if unexpected_directories:

        raise AssertionError(
            "Unexpected directories in publication directory:\n"
            + "\n".join(
                sorted(unexpected_directories)
            )
        )

    # Check split directory contents.
    expected_split_files = {
        TRAIN_IDS.name,
        VALIDATION_IDS.name,
        TEST_IDS.name,
        SPLIT_SUMMARY.name,
    }

    actual_split_files = {
        path.name
        for path in SPLIT_DIR.iterdir()
        if path.is_file()
    }

    unexpected_split_files = (
        actual_split_files
        - expected_split_files
    )

    if unexpected_split_files:

        raise AssertionError(
            "Unexpected files in split_ids directory:\n"
            + "\n".join(
                sorted(unexpected_split_files)
            )
        )

    logger.info(
        "PASS: no unexpected publication files found."
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():

    logger.info("=" * 70)
    logger.info("FINAL PUBLICATION PACKAGE CHECK")
    logger.info("=" * 70)

    check_required_files()

    dataset_ids = check_dataset()

    check_splits(dataset_ids)

    check_zip()

    check_checksums()

    check_no_unexpected_files()

    logger.info("=" * 70)
    logger.info("FINAL PUBLICATION PACKAGE CHECK PASSED")
    logger.info("=" * 70)

    logger.info(
        "The publication package is internally consistent."
    )

    logger.info(
        "No files were modified by this check."
    )


if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        logger.error("=" * 70)
        logger.error("FINAL PUBLICATION PACKAGE CHECK FAILED")
        logger.error("=" * 70)
        logger.error("%s", error)

        sys.exit(1)