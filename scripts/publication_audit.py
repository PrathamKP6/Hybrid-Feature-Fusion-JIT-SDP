"""
publication_audit.py

Audits the final publication/reproducibility dataset.

Input:
    data/public_release/
        final_multilingual_jit_codebert_llm_59996_768d.csv

Checks:
    - 59,996 rows
    - unique commit IDs
    - binary buggy labels
    - 12 JIT features
    - 768-D CodeBERT embeddings
    - project coverage when project column exists
    - language coverage when language column exists
    - chronological split sizes
    - no split overlap
    - complete split coverage
    - ZIP integrity

Output:
    data/public_release/AUDIT_REPORT.txt
"""

from pathlib import Path
import ast
import logging
import sys
import zipfile

import pandas as pd


# ============================================================
# PATHS
# ============================================================

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

SPLIT_DIR = (
    ROOT
    / "results"
    / "chronological_splits"
)

TRAIN_FILE = SPLIT_DIR / "train.csv"
VALIDATION_FILE = SPLIT_DIR / "validation.csv"
TEST_FILE = SPLIT_DIR / "test.csv"

AUDIT_REPORT = (
    PUBLIC_DIR
    / "AUDIT_REPORT.txt"
)


# ============================================================
# EXPECTED VALUES
# ============================================================

EXPECTED_ROWS = 59_996

EXPECTED_TRAIN = 41_998
EXPECTED_VALIDATION = 8_998
EXPECTED_TEST = 9_000

EXPECTED_EMBEDDING_DIM = 768

EXPECTED_PROJECTS = 19

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


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_buggy(value):

    if pd.isna(value):
        raise ValueError(
            "Missing buggy label found."
        )

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):

        if value == 0:
            return 0

        if value == 1:
            return 1

    text = str(value).strip().lower()

    if text in {"0", "false"}:
        return 0

    if text in {"1", "true"}:
        return 1

    raise ValueError(
        f"Invalid buggy label: {value!r}"
    )


# ============================================================
# EMBEDDING PARSER
# ============================================================

def parse_embedding(value):

    if pd.isna(value):
        return None

    if isinstance(value, (list, tuple)):
        return list(value)

    try:

        parsed = ast.literal_eval(
            str(value).strip()
        )

        if isinstance(parsed, (list, tuple)):
            return list(parsed)

    except Exception:
        return None

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info("=" * 70)
    logger.info(
        "PUBLICATION DATASET AUDIT"
    )
    logger.info("=" * 70)

    # ========================================================
    # LOAD DATASET
    # ========================================================

    if not DATASET.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET}"
        )

    logger.info(
        "Loading final dataset..."
    )

    df = pd.read_csv(
        DATASET,
        low_memory=False,
    )

    logger.info(
        "Shape: %s rows × %s columns",
        f"{len(df):,}",
        len(df.columns),
    )

    logger.info(
        "Columns in final dataset:"
    )

    for i, column in enumerate(
        df.columns,
        start=1,
    ):

        logger.info(
            "  %02d. %s",
            i,
            column,
        )

    # ========================================================
    # ROW COUNT
    # ========================================================

    if len(df) != EXPECTED_ROWS:

        raise AssertionError(
            f"Expected {EXPECTED_ROWS:,} rows, "
            f"found {len(df):,}"
        )

    logger.info(
        "PASS: row count = %s",
        f"{len(df):,}",
    )

    # ========================================================
    # COMMIT IDS
    # ========================================================

    if "commit_id" not in df.columns:

        raise AssertionError(
            "commit_id column missing."
        )

    missing_ids = int(
        df["commit_id"].isna().sum()
    )

    duplicate_ids = int(
        df["commit_id"].duplicated().sum()
    )

    if missing_ids:

        raise AssertionError(
            f"Missing commit IDs: {missing_ids}"
        )

    if duplicate_ids:

        raise AssertionError(
            f"Duplicate commit IDs: {duplicate_ids}"
        )

    logger.info(
        "PASS: 59,996 unique commit IDs."
    )

    # ========================================================
    # LABELS
    # ========================================================

    if "buggy" not in df.columns:

        raise AssertionError(
            "buggy column missing."
        )

    df["buggy"] = df["buggy"].apply(
        normalize_buggy
    )

    labels = sorted(
        df["buggy"].unique().tolist()
    )

    if labels != [0, 1]:

        raise AssertionError(
            f"Unexpected labels: {labels}"
        )

    non_buggy = int(
        (df["buggy"] == 0).sum()
    )

    buggy = int(
        (df["buggy"] == 1).sum()
    )

    logger.info(
        "Buggy distribution:"
    )

    logger.info(
        "  0 = %s",
        f"{non_buggy:,}",
    )

    logger.info(
        "  1 = %s",
        f"{buggy:,}",
    )

    logger.info(
        "PASS: binary buggy labels."
    )

    # ========================================================
    # JIT FEATURES
    # ========================================================

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
        "PASS: all 12 JIT features present."
    )

    # ========================================================
    # CODEBERT
    # ========================================================

    if "embedding" not in df.columns:

        raise AssertionError(
            "embedding column missing."
        )

    logger.info(
        "Checking 768-D CodeBERT embeddings..."
    )

    dimensions = set()
    invalid_embeddings = 0

    for value in df["embedding"]:

        embedding = parse_embedding(value)

        if embedding is None:

            invalid_embeddings += 1

            continue

        dimensions.add(
            len(embedding)
        )

    if invalid_embeddings:

        raise AssertionError(
            f"Invalid embeddings: "
            f"{invalid_embeddings:,}"
        )

    if dimensions != {
        EXPECTED_EMBEDDING_DIM
    }:

        raise AssertionError(
            "Unexpected embedding dimensions: "
            f"{dimensions}"
        )

    logger.info(
        "PASS: all embeddings are 768-dimensional."
    )

    # ========================================================
    # PROJECT AUDIT
    # ========================================================

    if "project" in df.columns:

        total_projects = (
            df["project"]
            .nunique()
        )

        logger.info(
            "Distinct projects: %d",
            total_projects,
        )

        if total_projects != EXPECTED_PROJECTS:

            raise AssertionError(
                f"Expected {EXPECTED_PROJECTS} "
                f"projects, found {total_projects}"
            )

        logger.info(
            "PASS: 19 projects."
        )

    else:

        logger.warning(
            "project column is not present "
            "in the final publication dataset."
        )

        logger.warning(
            "Project-level coverage will be "
            "verified from the frozen split/source "
            "datasets rather than the public master."
        )

    # ========================================================
    # LANGUAGE AUDIT
    # ========================================================

    if "language" in df.columns:

        logger.info(
            "Language column found; auditing it..."
        )

        logger.info(
            "Language distribution:"
        )

        language_counts = (
            df["language"]
            .value_counts()
            .sort_index()
        )

        for language, count in (
            language_counts.items()
        ):

            logger.info(
                "  %s = %s",
                language,
                f"{count:,}",
            )

        logger.info(
            "PASS: language metadata available."
        )

    else:

        logger.info(
            "Language column is not present "
            "in the final 44-column master dataset."
        )

        logger.info(
            "This is acceptable because language "
            "is not required as a model feature."
        )

    # ========================================================
    # MISSING VALUES
    # ========================================================

    logger.info(
        "Checking missing values..."
    )

    missing = (
        df.isna()
        .sum()
    )

    missing = missing[
        missing > 0
    ]

    if len(missing) == 0:

        logger.info(
            "PASS: no missing values."
        )

    else:

        logger.warning(
            "Missing values detected:"
        )

        for column, count in (
            missing.items()
        ):

            logger.warning(
                "  %s = %s",
                column,
                f"{count:,}",
            )

        logger.warning(
            "Review these values before publication."
        )

    # ========================================================
    # CHRONOLOGICAL SPLITS
    # ========================================================

    logger.info(
        "Checking chronological splits..."
    )

    split_files = {
        "train": (
            TRAIN_FILE,
            EXPECTED_TRAIN,
        ),
        "validation": (
            VALIDATION_FILE,
            EXPECTED_VALIDATION,
        ),
        "test": (
            TEST_FILE,
            EXPECTED_TEST,
        ),
    }

    split_ids = {}

    for split_name, (
        path,
        expected_count,
    ) in split_files.items():

        if not path.exists():

            raise FileNotFoundError(
                f"{split_name} split missing:\n{path}"
            )

        split_df = pd.read_csv(
            path,
            usecols=["commit_id"],
            low_memory=False,
        )

        if len(split_df) != expected_count:

            raise AssertionError(
                f"{split_name}: expected "
                f"{expected_count:,}, "
                f"found {len(split_df):,}"
            )

        if split_df["commit_id"].isna().any():

            raise AssertionError(
                f"{split_name}: missing IDs."
            )

        if split_df["commit_id"].duplicated().any():

            raise AssertionError(
                f"{split_name}: duplicate IDs."
            )

        split_ids[split_name] = set(
            split_df["commit_id"]
        )

        logger.info(
            "  %s = %s",
            split_name,
            f"{len(split_ids[split_name]):,}",
        )

    # ========================================================
    # SPLIT OVERLAP
    # ========================================================

    if (
        split_ids["train"]
        & split_ids["validation"]
    ):

        raise AssertionError(
            "Train/validation overlap found."
        )

    if (
        split_ids["train"]
        & split_ids["test"]
    ):

        raise AssertionError(
            "Train/test overlap found."
        )

    if (
        split_ids["validation"]
        & split_ids["test"]
    ):

        raise AssertionError(
            "Validation/test overlap found."
        )

    logger.info(
        "PASS: no train/validation/test overlap."
    )

    # ========================================================
    # SPLIT COVERAGE
    # ========================================================

    combined_split_ids = (
        split_ids["train"]
        | split_ids["validation"]
        | split_ids["test"]
    )

    final_ids = set(
        df["commit_id"]
    )

    if combined_split_ids != final_ids:

        missing = (
            final_ids
            - combined_split_ids
        )

        extra = (
            combined_split_ids
            - final_ids
        )

        raise AssertionError(
            "Split coverage mismatch.\n"
            f"Missing IDs: {len(missing):,}\n"
            f"Extra IDs: {len(extra):,}"
        )

    logger.info(
        "PASS: split IDs cover all "
        "59,996 commits."
    )

    # ========================================================
    # ZIP AUDIT
    # ========================================================

    if not ZIP_FILE.exists():

        raise FileNotFoundError(
            f"ZIP file missing:\n{ZIP_FILE}"
        )

    with zipfile.ZipFile(
        ZIP_FILE,
        "r",
    ) as archive:

        contents = archive.namelist()

    expected_zip = {
        DATASET.name
    }

    actual_zip = {
        Path(name).name
        for name in contents
    }

    if actual_zip != expected_zip:

        raise AssertionError(
            "ZIP contents are unexpected.\n"
            f"Expected: {expected_zip}\n"
            f"Found: {actual_zip}"
        )

    logger.info(
        "PASS: ZIP contains only final CSV."
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = []

    report.append(
        "PUBLICATION DATASET AUDIT REPORT"
    )

    report.append(
        "=" * 70
    )

    report.append("")
    report.append("STATUS: PASS")
    report.append("")

    report.append(
        f"Rows: {len(df):,}"
    )

    report.append(
        f"Columns: {len(df.columns)}"
    )

    report.append(
        f"Unique commit IDs: "
        f"{df['commit_id'].nunique():,}"
    )

    report.append(
        "CodeBERT dimension: 768"
    )

    report.append(
        "JIT features: 12"
    )

    if "project" in df.columns:

        report.append(
            f"Projects: "
            f"{df['project'].nunique():,}"
        )

    else:

        report.append(
            "Projects: not stored as a final-dataset column"
        )

    report.append("")

    report.append(
        "LABELS"
    )

    report.append(
        "-" * 70
    )

    report.append(
        f"Non-buggy (0): {non_buggy:,}"
    )

    report.append(
        f"Buggy (1): {buggy:,}"
    )

    report.append("")

    report.append(
        "SPLITS"
    )

    report.append(
        "-" * 70
    )

    report.append(
        f"Train: {len(split_ids['train']):,}"
    )

    report.append(
        f"Validation: "
        f"{len(split_ids['validation']):,}"
    )

    report.append(
        f"Test: {len(split_ids['test']):,}"
    )

    report.append("")

    report.append(
        "AUDITS"
    )

    report.append(
        "-" * 70
    )

    report.extend(
        [
            "[PASS] Row count",
            "[PASS] Unique commit IDs",
            "[PASS] Binary buggy labels",
            "[PASS] 12 JIT features",
            "[PASS] 768-D CodeBERT embeddings",
            "[PASS] Chronological split counts",
            "[PASS] No split overlap",
            "[PASS] Complete split coverage",
            "[PASS] ZIP integrity",
        ]
    )

    if "language" in df.columns:

        report.append(
            "[PASS] Language metadata"
        )

    else:

        report.append(
            "[INFO] Language column not stored "
            "in final master dataset"
        )

    if "project" in df.columns:

        report.append(
            "[PASS] Project metadata"
        )

    else:

        report.append(
            "[INFO] Project column not stored "
            "in final master dataset"
        )

    report.append("")
    report.append(
        "Publication audit completed."
    )

    AUDIT_REPORT.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    logger.info(
        "Audit report saved:"
    )

    logger.info(
        "%s",
        AUDIT_REPORT,
    )

    logger.info("=" * 70)
    logger.info(
        "PUBLICATION AUDIT PASSED"
    )
    logger.info("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        logger.error(
            "PUBLICATION AUDIT FAILED"
        )

        logger.error(
            "%s",
            error,
        )

        sys.exit(1)