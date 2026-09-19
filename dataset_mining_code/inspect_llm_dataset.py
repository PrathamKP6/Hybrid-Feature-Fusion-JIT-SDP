"""
Inspect the LLM-only preprocessed dataset.

Purpose
-------
Verify:
1. Train / validation / test sizes
2. Column names and data types
3. Target label distribution
4. Language distribution
5. Project distribution
6. Missing values
7. Duplicate rows / duplicate commit IDs
8. Embedding representation
9. Embedding dimensionality
10. Whether embeddings are 768D and whether PCA appears to have been applied
11. Train/validation/test ID overlap
12. Temporal ordering
13. Basic consistency across splits

Expected directory:
    data/
    └── llm_experiment_preprocessed/
        ├── train.csv
        ├── validation.csv
        ├── test.csv
        └── one_hot_encoder.joblib
"""

from pathlib import Path
import ast
import json
import sys

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/llm_experiment_preprocessed")

TRAIN_FILE = DATA_DIR / "train.csv"
VAL_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_subheader(title):
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


def load_csv(path):
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)

    print(f"Loading: {path}")
    df = pd.read_csv(path)
    print(f"Loaded shape: {df.shape}")
    return df


def inspect_basic(df, name):
    print_subheader(f"{name} BASIC INFORMATION")

    print(f"Rows       : {len(df):,}")
    print(f"Columns    : {len(df.columns):,}")
    print(f"Memory     : {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

    print("\nColumns:")
    for i, col in enumerate(df.columns):
        print(f"  [{i:3d}] {col}")

    print("\nData types:")
    print(df.dtypes.to_string())


def inspect_labels(df, name):
    print_subheader(f"{name} LABEL DISTRIBUTION")

    possible_labels = [
        "buggy",
        "label",
        "target",
        "y"
    ]

    label_col = None

    for col in possible_labels:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        print("[WARNING] Could not automatically identify label column.")
        return None

    print(f"Label column: {label_col}")

    counts = df[label_col].value_counts(dropna=False)
    percentages = df[label_col].value_counts(
        normalize=True,
        dropna=False
    ) * 100

    result = pd.DataFrame({
        "count": counts,
        "percentage": percentages.round(4)
    })

    print(result)

    return label_col


def inspect_languages(df, name):
    print_subheader(f"{name} LANGUAGE DISTRIBUTION")

    if "language" not in df.columns:
        print("[WARNING] No 'language' column found.")
        return

    counts = df["language"].value_counts(dropna=False)
    percentages = df["language"].value_counts(
        normalize=True,
        dropna=False
    ) * 100

    result = pd.DataFrame({
        "count": counts,
        "percentage": percentages.round(4)
    })

    print(result)


def inspect_projects(df, name):
    print_subheader(f"{name} PROJECT DISTRIBUTION")

    if "project" not in df.columns:
        print("[WARNING] No 'project' column found.")
        return

    counts = df["project"].value_counts(dropna=False)

    print(counts.to_string())

    print(f"\nNumber of unique projects: {df['project'].nunique(dropna=True)}")


def inspect_missing(df, name):
    print_subheader(f"{name} MISSING VALUES")

    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100

    result = pd.DataFrame({
        "missing_count": missing,
        "missing_percentage": missing_pct.round(4)
    })

    result = result[result["missing_count"] > 0]

    if len(result) == 0:
        print("No missing values found.")
    else:
        print(result.to_string())


def inspect_duplicates(df, name):
    print_subheader(f"{name} DUPLICATES")

    duplicate_rows = df.duplicated().sum()

    print(f"Duplicate complete rows: {duplicate_rows:,}")

    if "commit_id" in df.columns:
        duplicate_commit_ids = df["commit_id"].duplicated().sum()
        unique_commit_ids = df["commit_id"].nunique()

        print(f"Unique commit IDs       : {unique_commit_ids:,}")
        print(f"Duplicate commit IDs    : {duplicate_commit_ids:,}")

    else:
        print("[WARNING] No commit_id column found.")


# ============================================================
# EMBEDDING INSPECTION
# ============================================================

def detect_embedding_columns(df):
    """
    Detect whether CodeBERT embeddings are represented as:

    A) One column containing a list/string of 768 values
    B) 768 separate numeric columns
    C) Named embedding columns such as embedding_0 ... embedding_767
    """

    columns = list(df.columns)

    # --------------------------------------------------------
    # Case 1: obvious embedding column
    # --------------------------------------------------------

    possible_names = [
        "embedding",
        "embeddings",
        "codebert_embedding",
        "codebert_embeddings",
        "codebert",
        "vector"
    ]

    for col in possible_names:
        if col in columns:
            return {
                "type": "single_embedding_column",
                "columns": [col]
            }

    # --------------------------------------------------------
    # Case 2: columns containing embedding-like names
    # --------------------------------------------------------

    embedding_cols = []

    for col in columns:
        lower = str(col).lower()

        if (
            lower.startswith("embedding_")
            or lower.startswith("emb_")
            or lower.startswith("codebert_")
        ):
            embedding_cols.append(col)

    if len(embedding_cols) > 0:
        return {
            "type": "multiple_embedding_columns",
            "columns": embedding_cols
        }

    # --------------------------------------------------------
    # Case 3: detect large block of numeric columns
    # --------------------------------------------------------

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) >= 700:
        return {
            "type": "large_numeric_block",
            "columns": numeric_cols
        }

    return {
        "type": "not_detected",
        "columns": []
    }


def parse_embedding(value):
    """
    Convert common serialized embedding formats into numpy array.
    """

    if isinstance(value, np.ndarray):
        return value.astype(np.float32)

    if isinstance(value, list):
        return np.asarray(value, dtype=np.float32)

    if isinstance(value, tuple):
        return np.asarray(value, dtype=np.float32)

    if isinstance(value, str):

        value = value.strip()

        # Python-list format:
        # "[0.123, 0.456, ...]"
        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, (list, tuple)):
                return np.asarray(parsed, dtype=np.float32)

        except Exception:
            pass

        # Space-separated format:
        # "0.123 0.456 0.789 ..."
        try:
            values = np.fromstring(value, sep=" ")

            if len(values) > 0:
                return values.astype(np.float32)

        except Exception:
            pass

    return None


def inspect_embeddings(df, name):
    print_subheader(f"{name} CODEBERT EMBEDDING INSPECTION")

    info = detect_embedding_columns(df)

    print(f"Detected representation: {info['type']}")
    print(f"Detected columns: {len(info['columns'])}")

    # --------------------------------------------------------
    # Single embedding column
    # --------------------------------------------------------

    if info["type"] == "single_embedding_column":

        col = info["columns"][0]

        print(f"\nEmbedding column: {col}")

        sample_count = min(10, len(df))

        dimensions = []
        valid = 0
        invalid = 0

        for value in df[col].head(sample_count):

            vector = parse_embedding(value)

            if vector is None:
                invalid += 1
            else:
                valid += 1
                dimensions.append(len(vector))

        print(f"Valid sampled embeddings  : {valid}")
        print(f"Invalid sampled embeddings: {invalid}")

        if dimensions:
            print(f"Sampled dimensions: {dimensions}")
            print(f"Unique dimensions : {sorted(set(dimensions))}")

            if len(set(dimensions)) == 1:
                dim = dimensions[0]

                if dim == 768:
                    print("\n[OK] Embeddings appear to be 768-dimensional.")
                elif dim < 768:
                    print(
                        f"\n[INFO] Embeddings appear reduced to {dim} dimensions."
                    )
                else:
                    print(
                        f"\n[INFO] Embeddings appear to have {dim} dimensions."
                    )

        # Magnitude statistics
        norms = []

        for value in df[col].head(min(1000, len(df))):

            vector = parse_embedding(value)

            if vector is not None:
                norms.append(np.linalg.norm(vector))

        if norms:
            norms = np.asarray(norms)

            print("\nEmbedding norm statistics "
                  f"(first {len(norms):,} rows):")

            print(f"  Min    : {norms.min():.6f}")
            print(f"  Max    : {norms.max():.6f}")
            print(f"  Mean   : {norms.mean():.6f}")
            print(f"  Median : {np.median(norms):.6f}")
            print(f"  Std    : {norms.std():.6f}")

        return info

    # --------------------------------------------------------
    # Multiple embedding columns
    # --------------------------------------------------------

    if info["type"] in [
        "multiple_embedding_columns",
        "large_numeric_block"
    ]:

        print(f"\nNumber of detected embedding/numeric columns: "
              f"{len(info['columns'])}")

        print("\nFirst 20 detected columns:")
        for col in info["columns"][:20]:
            print(f"  {col}")

        print("\nLast 10 detected columns:")
        for col in info["columns"][-10:]:
            print(f"  {col}")

        if len(info["columns"]) == 768:
            print("\n[OK] Exactly 768 embedding columns detected.")
        elif len(info["columns"]) < 768:
            print(
                f"\n[INFO] Only {len(info['columns'])} embedding columns detected."
            )
        else:
            print(
                f"\n[INFO] {len(info['columns'])} embedding columns detected."
            )

        return info

    print(
        "\n[WARNING] Could not automatically detect CodeBERT embeddings."
    )

    print(
        "You should inspect the column names manually."
    )

    return info


# ============================================================
# ID OVERLAP
# ============================================================

def inspect_split_overlap(train, val, test):
    print_header("TRAIN / VALIDATION / TEST OVERLAP")

    if "commit_id" not in train.columns:
        print("[WARNING] commit_id not available.")
        return

    train_ids = set(train["commit_id"].dropna())
    val_ids = set(val["commit_id"].dropna())
    test_ids = set(test["commit_id"].dropna())

    train_val = train_ids & val_ids
    train_test = train_ids & test_ids
    val_test = val_ids & test_ids

    print(f"Train IDs:      {len(train_ids):,}")
    print(f"Validation IDs: {len(val_ids):,}")
    print(f"Test IDs:       {len(test_ids):,}")

    print("\nOverlaps:")
    print(f"Train ∩ Validation: {len(train_val):,}")
    print(f"Train ∩ Test      : {len(train_test):,}")
    print(f"Validation ∩ Test : {len(val_test):,}")

    if not train_val and not train_test and not val_test:
        print("\n[OK] No commit-ID overlap detected.")
    else:
        print("\n[WARNING] Commit-ID overlap detected!")


# ============================================================
# TEMPORAL ORDERING
# ============================================================

def inspect_dates(df, name):
    print_subheader(f"{name} TEMPORAL INFORMATION")

    possible_date_columns = [
        "author_date",
        "date",
        "commit_date",
        "timestamp"
    ]

    date_col = None

    for col in possible_date_columns:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        print("[WARNING] No date column detected.")
        return

    dates = pd.to_datetime(
        df[date_col],
        errors="coerce",
        utc=True
    )

    valid_dates = dates.dropna()

    print(f"Date column: {date_col}")
    print(f"Valid dates: {len(valid_dates):,}")
    print(f"Invalid dates: {dates.isna().sum():,}")

    if len(valid_dates) > 0:

        print(f"Earliest: {valid_dates.min()}")
        print(f"Latest  : {valid_dates.max()}")

        print("\nFirst 5 dates:")
        print(valid_dates.head().to_string(index=False))

        print("\nLast 5 dates:")
        print(valid_dates.tail().to_string(index=False))

        is_sorted = dates.is_monotonic_increasing

        print(f"\nSorted chronologically: {is_sorted}")

        if is_sorted:
            print("[OK] Split is chronologically sorted.")
        else:
            print(
                "[INFO] Split is not internally sorted. "
                "This may be normal depending on preprocessing."
            )


# ============================================================
# FEATURE / COLUMN CONSISTENCY
# ============================================================

def compare_columns(train, val, test):
    print_header("COLUMN CONSISTENCY")

    train_cols = set(train.columns)
    val_cols = set(val.columns)
    test_cols = set(test.columns)

    print(f"Train columns      : {len(train_cols)}")
    print(f"Validation columns : {len(val_cols)}")
    print(f"Test columns       : {len(test_cols)}")

    if train_cols == val_cols == test_cols:
        print("\n[OK] All three splits contain exactly the same columns.")
    else:

        print("\n[WARNING] Column mismatch detected.")

        print("\nTrain-only columns:")
        print(sorted(train_cols - val_cols - test_cols))

        print("\nValidation-only columns:")
        print(sorted(val_cols - train_cols - test_cols))

        print("\nTest-only columns:")
        print(sorted(test_cols - train_cols - val_cols))


# ============================================================
# DATASET SUMMARY
# ============================================================

def create_summary(train, val, test, label_col):
    print_header("OVERALL DATASET SUMMARY")

    rows = {
        "train": len(train),
        "validation": len(val),
        "test": len(test),
        "total": len(train) + len(val) + len(test)
    }

    print(json.dumps(rows, indent=4))

    if label_col:

        print("\nLabel distribution across splits:")

        for name, df in [
            ("train", train),
            ("validation", val),
            ("test", test)
        ]:

            counts = df[label_col].value_counts()

            total = len(df)

            print(f"\n{name}:")
            for label, count in counts.items():
                print(
                    f"  {label}: {count:,} "
                    f"({100 * count / total:.2f}%)"
                )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header("LLM-ONLY DATASET INSPECTION")

    print(f"Dataset directory: {DATA_DIR.resolve()}")

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    train = load_csv(TRAIN_FILE)
    validation = load_csv(VAL_FILE)
    test = load_csv(TEST_FILE)

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    inspect_basic(train, "TRAIN")
    inspect_basic(validation, "VALIDATION")
    inspect_basic(test, "TEST")

    # --------------------------------------------------------
    # Column consistency
    # --------------------------------------------------------

    compare_columns(train, validation, test)

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    train_label = inspect_labels(train, "TRAIN")
    inspect_labels(validation, "VALIDATION")
    inspect_labels(test, "TEST")

    # --------------------------------------------------------
    # Languages
    # --------------------------------------------------------

    inspect_languages(train, "TRAIN")
    inspect_languages(validation, "VALIDATION")
    inspect_languages(test, "TEST")

    # --------------------------------------------------------
    # Projects
    # --------------------------------------------------------

    inspect_projects(train, "TRAIN")
    inspect_projects(validation, "VALIDATION")
    inspect_projects(test, "TEST")

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    inspect_missing(train, "TRAIN")
    inspect_missing(validation, "VALIDATION")
    inspect_missing(test, "TEST")

    # --------------------------------------------------------
    # Duplicates
    # --------------------------------------------------------

    inspect_duplicates(train, "TRAIN")
    inspect_duplicates(validation, "VALIDATION")
    inspect_duplicates(test, "TEST")

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    train_embedding_info = inspect_embeddings(
        train,
        "TRAIN"
    )

    validation_embedding_info = inspect_embeddings(
        validation,
        "VALIDATION"
    )

    test_embedding_info = inspect_embeddings(
        test,
        "TEST"
    )

    # --------------------------------------------------------
    # Cross-split overlap
    # --------------------------------------------------------

    inspect_split_overlap(
        train,
        validation,
        test
    )

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    inspect_dates(train, "TRAIN")
    inspect_dates(validation, "VALIDATION")
    inspect_dates(test, "TEST")

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    create_summary(
        train,
        validation,
        test,
        train_label
    )

    # --------------------------------------------------------
    # Final conclusion
    # --------------------------------------------------------

    print_header("INSPECTION COMPLETE")

    print(
        """
Next step:
    Save/copy the complete output of this script.

Do NOT apply PCA, modify labels, resample, or regenerate embeddings yet.

We first need to confirm:
    1. Exact embedding representation
    2. Exact embedding dimension
    3. Label distribution
    4. Language distribution
    5. Split sizes
    6. No commit overlap
    7. Temporal ordering
    8. Whether these files are compatible with your existing experiments
"""
    )


if __name__ == "__main__":
    main()