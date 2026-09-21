from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

SOURCE_768 = DATA_DIR / "final_multilanguage_dataset_with_embeddings.csv"
PYCPP_768 = DATA_DIR / "python_cpp_23996_embeddings_768d.csv"

OUTPUT_768 = DATA_DIR / "final_multilanguage_59996_768d_reconstructed.csv"


# ============================================================
# CONFIGURATION
# ============================================================

JAVA_ROWS = 36_000
PYCPP_ROWS = 23_996
EXPECTED_TOTAL = JAVA_ROWS + PYCPP_ROWS


print("=" * 90)
print("RECONSTRUCTING 59,996-ROW 768D MULTILANGUAGE DATASET")
print("=" * 90)

print()
print("Input Java-containing 768D file:")
print(SOURCE_768)

print()
print("Input Python/C++ 768D file:")
print(PYCPP_768)

print()
print("Output:")
print(OUTPUT_768)

print()


# ============================================================
# CHECK INPUT FILES
# ============================================================

if not SOURCE_768.exists():
    raise FileNotFoundError(
        f"Java-containing source file not found:\n{SOURCE_768}"
    )

if not PYCPP_768.exists():
    raise FileNotFoundError(
        f"Python/C++ file not found:\n{PYCPP_768}"
    )


# ============================================================
# LOAD
# ============================================================

print("=" * 90)
print("LOADING INPUT DATASETS")
print("=" * 90)

source = pd.read_csv(
    SOURCE_768,
    low_memory=False
)

pycpp = pd.read_csv(
    PYCPP_768,
    low_memory=False
)

print(f"Source dataset : {len(source):,} rows × {len(source.columns):,} columns")
print(f"Py/C++ dataset : {len(pycpp):,} rows × {len(pycpp.columns):,} columns")

print()


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = [
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
    "embedding",
]

for col in required_columns:

    if col not in source.columns:
        raise ValueError(
            f"Required column '{col}' missing from source dataset."
        )

    if col not in pycpp.columns:
        raise ValueError(
            f"Required column '{col}' missing from Py/C++ dataset."
        )


# ============================================================
# IDENTIFY JAVA ROWS
# ============================================================

print("=" * 90)
print("IDENTIFYING JAVA ROWS")
print("=" * 90)

# Java projects in the source dataset are Apache Java projects.
# We identify Java rows using the project naming convention
# present in the source dataset.

java_mask = source["project"].astype(str).str.startswith("apache/")

java = source.loc[java_mask].copy()

print(f"Total source rows : {len(source):,}")
print(f"Java candidate rows: {len(java):,}")

print()


# ============================================================
# SAFETY CHECK
# ============================================================

if len(java) < JAVA_ROWS:

    raise ValueError(
        f"Only {len(java):,} Java candidate rows found, "
        f"but {JAVA_ROWS:,} are required."
    )


# ============================================================
# PARSE AUTHOR DATE
# ============================================================

print("=" * 90)
print("PARSING AUTHOR DATE")
print("=" * 90)

java["author_date"] = pd.to_datetime(
    java["author_date"],
    errors="raise"
)

print("author_date parsed successfully.")
print()


# ============================================================
# EXACT JAVA SELECTION LOGIC
# ============================================================

print("=" * 90)
print("SELECTING NEWEST 36,000 JAVA COMMITS")
print("=" * 90)

print("Sorting using:")
print('  ["author_date", "commit_id"]')
print("ascending:")
print("  [False, False]")
print()

java = java.sort_values(
    ["author_date", "commit_id"],
    ascending=[False, False]
)

java_36k = java.head(JAVA_ROWS).copy()

print(f"Selected Java rows: {len(java_36k):,}")

print()
print("Selected Java date range:")

print(
    "  Newest:",
    java_36k["author_date"].max()
)

print(
    "  Oldest:",
    java_36k["author_date"].min()
)

print()
print("Boundary Java commit IDs:")

print(
    "  First:",
    java_36k.iloc[0]["commit_id"]
)

print(
    "  Last:",
    java_36k.iloc[-1]["commit_id"]
)

print()


# ============================================================
# JAVA DUPLICATE CHECK
# ============================================================

java_duplicate_ids = java_36k["commit_id"].duplicated().sum()

print("=" * 90)
print("JAVA VALIDATION")
print("=" * 90)

print(f"Java rows             : {len(java_36k):,}")
print(f"Unique commit IDs     : {java_36k['commit_id'].nunique():,}")
print(f"Duplicate commit IDs  : {java_duplicate_ids:,}")

if java_duplicate_ids != 0:
    raise ValueError(
        "Duplicate commit IDs found in selected Java rows."
    )

print()


# ============================================================
# PYTHON/C++ VALIDATION
# ============================================================

print("=" * 90)
print("PYTHON/C++ VALIDATION")
print("=" * 90)

print(f"Rows                : {len(pycpp):,}")
print(f"Unique commit IDs   : {pycpp['commit_id'].nunique():,}")
print(
    f"Duplicate commit IDs: "
    f"{pycpp['commit_id'].duplicated().sum():,}"
)

if len(pycpp) != PYCPP_ROWS:

    raise ValueError(
        f"Expected {PYCPP_ROWS:,} Py/C++ rows, "
        f"found {len(pycpp):,}."
    )

if pycpp["commit_id"].duplicated().any():

    raise ValueError(
        "Duplicate commit IDs found in Py/C++ dataset."
    )

print()


# ============================================================
# CROSS-DATASET DUPLICATE CHECK
# ============================================================

print("=" * 90)
print("CROSS-DATASET COMMIT ID CHECK")
print("=" * 90)

java_ids = set(java_36k["commit_id"].astype(str))
pycpp_ids = set(pycpp["commit_id"].astype(str))

overlap = java_ids.intersection(pycpp_ids)

print(f"Java commit IDs       : {len(java_ids):,}")
print(f"Py/C++ commit IDs     : {len(pycpp_ids):,}")
print(f"Cross-dataset overlap : {len(overlap):,}")

if overlap:

    print()
    print("WARNING: overlapping commit IDs detected.")

    print("First 10 overlapping IDs:")

    for commit_id in list(overlap)[:10]:
        print(" ", commit_id)

    raise ValueError(
        "Java and Py/C++ datasets contain overlapping commit IDs."
    )

print()


# ============================================================
# KEEP COMMON SCHEMA
# ============================================================

print("=" * 90)
print("ALIGNING SCHEMA")
print("=" * 90)

common_columns = [
    col
    for col in pycpp.columns
    if col in java_36k.columns
]

print(f"Common columns: {len(common_columns)}")

print()
print("Columns retained:")

for i, col in enumerate(common_columns):
    print(f"  {i:2d}: {col}")

print()


# ============================================================
# VERIFY EMBEDDING DIMENSION
# ============================================================

import ast


def embedding_dimension(value):

    parsed = ast.literal_eval(str(value))

    return len(parsed)


print("=" * 90)
print("VERIFYING EMBEDDING DIMENSIONS")
print("=" * 90)

java_embedding_dim = embedding_dimension(
    java_36k["embedding"].iloc[0]
)

pycpp_embedding_dim = embedding_dimension(
    pycpp["embedding"].iloc[0]
)

print(f"Java embedding dimension  : {java_embedding_dim}")
print(f"Py/C++ embedding dimension: {pycpp_embedding_dim}")

if java_embedding_dim != 768:

    raise ValueError(
        f"Java embedding dimension is {java_embedding_dim}, expected 768."
    )

if pycpp_embedding_dim != 768:

    raise ValueError(
        f"Py/C++ embedding dimension is {pycpp_embedding_dim}, expected 768."
    )

print()
print("Both datasets contain 768-dimensional embeddings.")
print()


# ============================================================
# PREPARE DATASETS
# ============================================================

java_final = java_36k[common_columns].copy()

pycpp_final = pycpp[common_columns].copy()


# ============================================================
# CONCATENATE
# ============================================================

print("=" * 90)
print("CONCATENATING")
print("=" * 90)

print("Order:")
print("  1. Java 36,000")
print("  2. Python/C++ 23,996")
print()

final_768 = pd.concat(
    [
        java_final,
        pycpp_final
    ],
    axis=0,
    ignore_index=True
)

print(f"Combined rows: {len(final_768):,}")
print(f"Combined columns: {len(final_768.columns):,}")

print()


# ============================================================
# FINAL VALIDATION
# ============================================================

print("=" * 90)
print("FINAL VALIDATION")
print("=" * 90)

if len(final_768) != EXPECTED_TOTAL:

    raise ValueError(
        f"Expected {EXPECTED_TOTAL:,} rows, "
        f"got {len(final_768):,}."
    )

if final_768["commit_id"].nunique() != EXPECTED_TOTAL:

    raise ValueError(
        "Final dataset contains duplicate commit IDs."
    )


# Verify Java section
final_java_ids = set(
    final_768.iloc[:JAVA_ROWS]["commit_id"].astype(str)
)

if final_java_ids != java_ids:

    raise ValueError(
        "Final Java section does not match selected Java rows."
    )


# Verify Py/C++ section
final_pycpp_ids = set(
    final_768.iloc[JAVA_ROWS:]["commit_id"].astype(str)
)

if final_pycpp_ids != pycpp_ids:

    raise ValueError(
        "Final Py/C++ section does not match input Py/C++ rows."
    )


print("PASS: Total rows = 59,996")
print("PASS: Java rows = 36,000")
print("PASS: Py/C++ rows = 23,996")
print("PASS: Unique commit IDs = 59,996")
print("PASS: Java section preserved")
print("PASS: Py/C++ section preserved")
print("PASS: 768D embeddings verified")

print()


# ============================================================
# SAVE NEW DATASET
# ============================================================

print("=" * 90)
print("SAVING")
print("=" * 90)

if OUTPUT_768.exists():

    raise FileExistsError(
        f"\nOutput already exists:\n{OUTPUT_768}\n\n"
        "Refusing to overwrite it."
    )

final_768.to_csv(
    OUTPUT_768,
    index=False
)

print("Created:")
print(OUTPUT_768)

print()
print("=" * 90)
print("RECONSTRUCTION COMPLETE")
print("=" * 90)

print(f"Rows       : {len(final_768):,}")
print(f"Columns    : {len(final_768.columns):,}")
print("Embeddings : 768D")
print("Java       : 36,000")
print("Python/C++ : 23,996")
print("Total      : 59,996")
print()
print("Original input files were NOT modified.")
print("=" * 90)