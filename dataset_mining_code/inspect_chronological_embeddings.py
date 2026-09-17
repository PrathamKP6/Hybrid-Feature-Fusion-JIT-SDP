"""
Inspect CodeBERT embeddings in the canonical chronological splits.

Purpose
-------
Determine exactly how CodeBERT embeddings are stored in:

    results/chronological_splits/
        train.csv
        validation.csv
        test.csv

Checks:
    1. File sizes
    2. Row counts
    3. Column names
    4. Potential embedding columns
    5. Embedding representation
    6. Embedding dimensionality
    7. Whether dimensions are consistent across splits
    8. Numeric feature count
    9. Buggy-label representation
   10. Missing values
   11. Basic train/validation/test consistency

IMPORTANT:
    This script does NOT:
        - apply PCA
        - modify the datasets
        - rewrite CSVs
        - remove rows
        - alter labels

It is read-only.
"""

from pathlib import Path
import ast
import csv
import os
import sys

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SPLIT_DIR = Path("results/pca")

FILES = {
    "train": SPLIT_DIR / "train_pca384.csv",
    "validation": SPLIT_DIR / "validation_pca384.csv",
    "test": SPLIT_DIR / "test_pca384.csv",
}

# Number of rows used for embedding inspection
SAMPLE_ROWS = 20

# Number of rows used for lightweight statistics
STATS_ROWS = 1000

# Chunk size for reading large files
CHUNK_SIZE = 10_000


# ============================================================
# PRINT HELPERS
# ============================================================

def header(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def section(title):
    print("\n" + "-" * 75)
    print(title)
    print("-" * 75)


# ============================================================
# FILE INFORMATION
# ============================================================

def inspect_file(path, name):

    section(f"{name.upper()} FILE")

    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        return None

    size_gb = path.stat().st_size / (1024 ** 3)
    size_mb = path.stat().st_size / (1024 ** 2)

    print(f"Path      : {path}")
    print(f"Size      : {size_gb:.3f} GB ({size_mb:.1f} MB)")

    return path


# ============================================================
# HEADER INSPECTION
# ============================================================

def read_header(path):

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as f:

        reader = csv.reader(f)
        columns = next(reader)

    return columns


# ============================================================
# EMBEDDING COLUMN DETECTION
# ============================================================

def detect_embedding_columns(columns):

    embedding_cols = []

    for col in columns:

        c = str(col).lower().strip()

        if (
            "embedding" in c
            or "codebert" in c
            or c.startswith("emb_")
            or c.startswith("emb")
        ):
            embedding_cols.append(col)

    return embedding_cols


def print_columns(columns):

    print(f"Total columns: {len(columns)}")

    for i, col in enumerate(columns):

        print(f"  [{i:4d}] {col}")


# ============================================================
# SERIALIZED VECTOR PARSER
# ============================================================

def parse_vector(value):

    if value is None:
        return None

    if isinstance(value, np.ndarray):
        return value.astype(np.float32)

    if isinstance(value, (list, tuple)):
        try:
            return np.asarray(
                value,
                dtype=np.float32
            )
        except Exception:
            return None

    if not isinstance(value, str):
        return None

    value = value.strip()

    if not value:
        return None

    # --------------------------------------------------------
    # Python-list representation
    # --------------------------------------------------------

    try:

        parsed = ast.literal_eval(value)

        if isinstance(parsed, (list, tuple)):

            arr = np.asarray(
                parsed,
                dtype=np.float32
            )

            if arr.ndim == 1:
                return arr

    except Exception:
        pass

    # --------------------------------------------------------
    # Space-separated representation
    # --------------------------------------------------------

    try:

        arr = np.fromstring(
            value,
            sep=" ",
            dtype=np.float32
        )

        if len(arr) > 0:
            return arr

    except Exception:
        pass

    # --------------------------------------------------------
    # Comma-separated representation
    # --------------------------------------------------------

    try:

        arr = np.fromstring(
            value,
            sep=",",
            dtype=np.float32
        )

        if len(arr) > 0:
            return arr

    except Exception:
        pass

    return None


# ============================================================
# SAMPLE DATA
# ============================================================

def read_sample(path):

    try:

        df = pd.read_csv(
            path,
            nrows=SAMPLE_ROWS,
            low_memory=False
        )

        return df

    except Exception as e:

        print(f"[ERROR] Could not read sample: {e}")
        return None


# ============================================================
# EMBEDDING INSPECTION
# ============================================================

def inspect_embeddings(path, name):

    section(f"{name.upper()} EMBEDDING INSPECTION")

    df = read_sample(path)

    if df is None:
        return None

    columns = list(df.columns)

    embedding_cols = detect_embedding_columns(columns)

    print(
        f"Embedding-like columns detected: "
        f"{len(embedding_cols)}"
    )

    if embedding_cols:

        print("\nEmbedding-like columns:")

        for col in embedding_cols[:30]:
            print(f"  {col}")

        if len(embedding_cols) > 30:
            print(
                f"  ... and "
                f"{len(embedding_cols) - 30} more"
            )

    # --------------------------------------------------------
    # CASE 1: Many embedding columns
    # --------------------------------------------------------

    if len(embedding_cols) >= 100:

        print("\nRepresentation:")
        print("  Multiple embedding columns")

        print(
            f"\nEmbedding dimensionality = "
            f"{len(embedding_cols)}"
        )

        if len(embedding_cols) == 768:
            print("[OK] 768D CodeBERT representation detected.")

        elif len(embedding_cols) == 384:
            print(
                "[INFO] 384D representation detected. "
                "This may indicate PCA-reduced embeddings."
            )

        else:
            print(
                "[INFO] Embedding dimensionality is "
                f"{len(embedding_cols)}."
            )

        return {
            "type": "multiple_columns",
            "dimension": len(embedding_cols),
            "columns": embedding_cols,
        }

    # --------------------------------------------------------
    # CASE 2: One serialized embedding column
    # --------------------------------------------------------

    if len(embedding_cols) == 1:

        col = embedding_cols[0]

        print("\nRepresentation:")
        print("  Single serialized embedding column")

        print(f"Embedding column: {col}")

        dimensions = []
        valid = 0
        invalid = 0

        for value in df[col].head(SAMPLE_ROWS):

            vector = parse_vector(value)

            if vector is None:

                invalid += 1

            else:

                valid += 1
                dimensions.append(len(vector))

        print(f"\nValid vectors  : {valid}")
        print(f"Invalid vectors: {invalid}")

        print(
            f"Detected dimensions: "
            f"{sorted(set(dimensions))}"
        )

        if dimensions:

            unique_dims = sorted(set(dimensions))

            if len(unique_dims) == 1:

                dimension = unique_dims[0]

                print(
                    f"\nEmbedding dimensionality = "
                    f"{dimension}"
                )

                if dimension == 768:

                    print(
                        "[OK] 768D CodeBERT representation detected."
                    )

                elif dimension == 384:

                    print(
                        "[INFO] 384D representation detected. "
                        "This may indicate PCA reduction."
                    )

                else:

                    print(
                        "[INFO] Non-standard embedding dimension."
                    )

                return {
                    "type": "serialized",
                    "dimension": dimension,
                    "column": col,
                }

    # --------------------------------------------------------
    # CASE 3: No obvious embedding column
    # --------------------------------------------------------

    print(
        "\n[WARNING] No obvious CodeBERT embedding "
        "representation was detected."
    )

    # Look for large numeric blocks
    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    print(
        f"\nNumeric columns in sample: "
        f"{len(numeric_cols)}"
    )

    if len(numeric_cols) >= 700:

        print(
            "[INFO] Large numeric block detected."
        )

        print(
            f"Potential embedding dimension: "
            f"{len(numeric_cols)}"
        )

        return {
            "type": "numeric_block",
            "dimension": len(numeric_cols),
            "columns": numeric_cols,
        }

    return None


# ============================================================
# BUGGY LABEL INSPECTION
# ============================================================

def inspect_buggy(path, name):

    section(f"{name.upper()} BUGGY LABEL INSPECTION")

    try:

        chunks = pd.read_csv(
            path,
            usecols=lambda c: str(c).strip().lower() == "buggy",
            chunksize=CHUNK_SIZE,
            low_memory=False
        )

    except Exception as e:

        print(f"[ERROR] Could not read buggy column: {e}")
        return

    counts = {}

    total = 0

    for chunk in chunks:

        col = chunk.columns[0]

        for value in chunk[col]:

            # Normalize representation only for inspection.
            # The actual dataset is NOT modified.

            if pd.isna(value):

                key = "<NA>"

            else:

                key = str(value).strip()

            counts[key] = counts.get(key, 0) + 1
            total += 1

    print(f"Total buggy values: {total:,}")

    print("\nRaw label representations:")

    for value, count in sorted(
        counts.items(),
        key=lambda x: str(x[0])
    ):

        pct = count / total * 100

        print(
            f"  {repr(value):15s} "
            f"{count:10,} "
            f"({pct:8.4f}%)"
        )


# ============================================================
# ROW COUNT
# ============================================================

def count_rows(path):

    # Efficient CSV line count.
    # Subtract header.

    with open(
        path,
        "rb"
    ) as f:

        count = sum(
            1 for _ in f
        )

    return max(count - 1, 0)


# ============================================================
# MISSING VALUE INSPECTION
# ============================================================

def inspect_missing(path, name):

    section(f"{name.upper()} MISSING VALUE SAMPLE")

    try:

        df = pd.read_csv(
            path,
            nrows=STATS_ROWS,
            low_memory=False
        )

        missing = df.isna().sum()

        missing = missing[
            missing > 0
        ].sort_values(
            ascending=False
        )

        if len(missing) == 0:

            print(
                f"No missing values in first "
                f"{len(df):,} rows."
            )

        else:

            print(
                f"Missing values in first "
                f"{len(df):,} rows:"
            )

            print(missing.to_string())

    except Exception as e:

        print(f"[WARNING] Missing-value check failed: {e}")


# ============================================================
# CROSS-SPLIT COLUMN CONSISTENCY
# ============================================================

def inspect_column_consistency(all_columns):

    header("COLUMN CONSISTENCY")

    names = list(all_columns.keys())

    reference = all_columns[names[0]]

    all_same = True

    for name in names[1:]:

        if all_columns[name] != reference:

            all_same = False

            print(
                f"[WARNING] {name} columns differ "
                f"from {names[0]}."
            )

    if all_same:

        print(
            "[OK] Train, validation and test "
            "have identical column order."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    header(
        "CANONICAL CHRONOLOGICAL SPLIT "
        "CODEBERT / PCA INSPECTION"
    )

    print(f"Directory: {SPLIT_DIR.resolve()}")

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    valid_files = {}

    for name, path in FILES.items():

        result = inspect_file(
            path,
            name
        )

        if result is not None:
            valid_files[name] = path

    if len(valid_files) != 3:

        print(
            "\n[ERROR] One or more split files are missing."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Count rows
    # --------------------------------------------------------

    header("ROW COUNTS")

    row_counts = {}

    for name, path in valid_files.items():

        print(
            f"Counting {name} rows..."
        )

        count = count_rows(path)

        row_counts[name] = count

        print(
            f"{name:12s}: {count:,}"
        )

    print(
        f"\nTotal: "
        f"{sum(row_counts.values()):,}"
    )

    # --------------------------------------------------------
    # Read headers
    # --------------------------------------------------------

    all_columns = {}

    header("COLUMN STRUCTURE")

    for name, path in valid_files.items():

        columns = read_header(path)

        all_columns[name] = columns

        print(
            f"\n{name.upper()}: "
            f"{len(columns)} columns"
        )

        print_columns(columns)

    # --------------------------------------------------------
    # Compare columns
    # --------------------------------------------------------

    inspect_column_consistency(
        all_columns
    )

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    embedding_results = {}

    for name, path in valid_files.items():

        embedding_results[name] = inspect_embeddings(
            path,
            name
        )

    # --------------------------------------------------------
    # Compare embedding dimensions
    # --------------------------------------------------------

    header("EMBEDDING DIMENSION CONSISTENCY")

    dimensions = {}

    for name, result in embedding_results.items():

        if result is None:

            dimensions[name] = None

            print(
                f"{name:12s}: NOT DETECTED"
            )

        else:

            dimensions[name] = result["dimension"]

            print(
                f"{name:12s}: "
                f"{result['dimension']}D"
            )

    detected_dims = [
        x for x in dimensions.values()
        if x is not None
    ]

    if detected_dims:

        if len(set(detected_dims)) == 1:

            print(
                f"\n[OK] All detected splits use "
                f"{detected_dims[0]}D embeddings."
            )

        else:

            print(
                "\n[WARNING] Embedding dimensions "
                "differ across splits."
            )

    # --------------------------------------------------------
    # Buggy
    # --------------------------------------------------------

    for name, path in valid_files.items():

        inspect_buggy(
            path,
            name
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    for name, path in valid_files.items():

        inspect_missing(
            path,
            name
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    header("FINAL INTERPRETATION")

    print("""
Do NOT modify the datasets based on this inspection.

We are looking for:

    A. 768D CodeBERT embeddings
       -> raw CodeBERT representation

    B. 384D CodeBERT embeddings
       -> likely PCA-reduced representation

    C. Another dimensionality
       -> investigate before experiments

    D. Inconsistent dimensions across splits
       -> stop and investigate

    E. Mixed buggy representations
       -> normalize safely inside experiment code,
          without altering the source CSVs

After this inspection we can proceed to the
remaining experiments using the canonical splits.
""")


if __name__ == "__main__":
    main()