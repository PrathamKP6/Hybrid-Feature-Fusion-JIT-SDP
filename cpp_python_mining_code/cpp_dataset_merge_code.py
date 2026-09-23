import pandas as pd
from pathlib import Path
import sys


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_FILES = {
    "GODOT": "cpp_godot_dataset.csv",
    "KRITA": "cpp_krita_dataset.csv",
    "OPENCV": "cpp_opencv_dataset.csv",
}

OUTPUT_FILE = "final_cpp_12k.csv"
KRITA_EXTRA_FILE = "krita_extra_rows.csv"

EXPECTED_TOTAL_ROWS = 12000

# Column used to determine chronological age of commits
DATE_COLUMN = "author_date"


# ============================================================
# EXPECTED FINAL SCHEMA
# ============================================================

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
]


# ============================================================
# EXPECTED PROJECT VALUES
# ============================================================

EXPECTED_PROJECTS = {
    "GODOT": "cpp/godot",
    "KRITA": "cpp/krita",
    "OPENCV": "cpp/opencv",
}


# ============================================================
# IMPORTANT COLUMNS
# ============================================================

IMPORTANT_COLUMNS = [
    "commit_id",
    "project",
    "buggy",
    "fix",
    "year",
    "author_date",
]


# ============================================================
# ANOMALY STORAGE
# ============================================================

anomalies = []


def report(dataset, message):
    anomalies.append(
        f"[{dataset}] {message}"
    )


# ============================================================
# FILE EXISTENCE
# ============================================================

def check_file_exists(path):

    if not Path(path).exists():

        report(
            "FILES",
            f"Missing file: {path}"
        )

        return False

    return True


# ============================================================
# DUPLICATE COLUMN CHECK
# ============================================================

def check_duplicate_columns(df, dataset):

    duplicated = df.columns[
        df.columns.duplicated()
    ].tolist()

    if duplicated:

        report(
            dataset,
            f"Duplicate column names found: "
            f"{duplicated}"
        )


# ============================================================
# SCHEMA CHECK
# ============================================================

def check_schema(df, dataset):

    actual = list(df.columns)

    missing = [
        col
        for col in EXPECTED_COLUMNS
        if col not in actual
    ]

    extra = [
        col
        for col in actual
        if col not in EXPECTED_COLUMNS
    ]

    if missing:

        report(
            dataset,
            f"Missing columns: {missing}"
        )

    if extra:

        report(
            dataset,
            f"Unexpected columns: {extra}"
        )

    if not missing and not extra:

        if actual != EXPECTED_COLUMNS:

            report(
                dataset,
                "Column order is incorrect.\n"
                f"Expected: {EXPECTED_COLUMNS}\n"
                f"Found:    {actual}"
            )


# ============================================================
# PROJECT NORMALIZATION
# ============================================================

def normalize_project(
    df,
    dataset,
    expected_project
):

    if "project" not in df.columns:

        report(
            dataset,
            "Missing 'project' column."
        )

        return df

    existing_values = set(
        df["project"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )

    null_count = df["project"].isna().sum()

    if null_count > 0:

        report(
            dataset,
            f"'project' contains "
            f"{null_count} null values."
        )

    empty_count = (
        df["project"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if empty_count > 0:

        report(
            dataset,
            f"'project' contains "
            f"{empty_count} empty-string values."
        )

    short_project = expected_project.replace(
        "cpp/",
        ""
    )

    accepted_values = {
        short_project,
        expected_project,
    }

    unexpected = (
        existing_values
        - accepted_values
    )

    if unexpected:

        report(
            dataset,
            f"Unexpected project values: "
            f"{sorted(unexpected)}. "
            f"Expected '{short_project}' "
            f"or '{expected_project}'."
        )

        return df

    # Normalize to canonical value
    df["project"] = expected_project

    return df


# ============================================================
# DUPLICATE COMMIT ID CHECK
# ============================================================

def check_duplicate_commit_ids(
    df,
    dataset
):

    if "commit_id" not in df.columns:
        return

    duplicated_mask = df[
        "commit_id"
    ].duplicated(keep=False)

    duplicates = df[
        duplicated_mask
    ]

    if not duplicates.empty:

        duplicate_ids = (
            duplicates["commit_id"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        report(
            dataset,
            f"Duplicate commit_id values found: "
            f"{len(duplicate_ids)} unique duplicate IDs. "
            f"Examples: {duplicate_ids[:10]}"
        )


# ============================================================
# NULL CHECK
# ============================================================

def check_nulls(df, dataset):

    for col in IMPORTANT_COLUMNS:

        if col not in df.columns:
            continue

        null_count = df[col].isna().sum()

        if null_count > 0:

            report(
                dataset,
                f"Column '{col}' contains "
                f"{null_count} null values."
            )


# ============================================================
# EMPTY STRING CHECK
# ============================================================

def check_empty_strings(
    df,
    dataset
):

    for col in [
        "commit_id",
        "project",
        "author_date",
    ]:

        if col not in df.columns:
            continue

        empty_count = (
            df[col]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        if empty_count > 0:

            report(
                dataset,
                f"Column '{col}' contains "
                f"{empty_count} empty-string values."
            )


# ============================================================
# BUG / FIX BINARY CHECK
# ============================================================

def check_binary_columns(
    df,
    dataset
):

    for col in [
        "buggy",
        "fix",
    ]:

        if col not in df.columns:
            continue

        values = set(
            df[col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .unique()
        )

        allowed = {
            "0",
            "1",
            "0.0",
            "1.0",
            "false",
            "true",
        }

        invalid = values - allowed

        if invalid:

            report(
                dataset,
                f"Column '{col}' contains "
                f"unexpected values: "
                f"{sorted(invalid)}"
            )


# ============================================================
# NUMERIC COLUMN CHECK
# ============================================================

def check_numeric_columns(
    df,
    dataset
):

    numeric_columns = [
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
    ]

    for col in numeric_columns:

        if col not in df.columns:
            continue

        converted = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        invalid = (
            converted.isna()
            & df[col].notna()
        )

        invalid_count = invalid.sum()

        if invalid_count > 0:

            report(
                dataset,
                f"Column '{col}' contains "
                f"{invalid_count} "
                f"non-numeric values."
            )


# ============================================================
# DATE CHECK
# ============================================================

def check_date_column(
    df,
    dataset
):

    if DATE_COLUMN not in df.columns:
        return

    parsed_dates = pd.to_datetime(
        df[DATE_COLUMN],
        errors="coerce"
    )

    invalid_count = (
        parsed_dates.isna()
        & df[DATE_COLUMN].notna()
    ).sum()

    if invalid_count > 0:

        report(
            dataset,
            f"Column '{DATE_COLUMN}' contains "
            f"{invalid_count} invalid/unparseable "
            f"date values."
        )


# ============================================================
# ROW COUNT CHECK
# ============================================================

def check_row_count(
    df,
    dataset
):

    if len(df) == 0:

        report(
            dataset,
            "Dataset contains 0 rows."
        )


# ============================================================
# LOAD FILES
# ============================================================

print("=" * 70)
print("FINAL C++ DATASET VALIDATION")
print("=" * 70)


for dataset, filename in PROJECT_FILES.items():

    print(
        f"\nChecking file: {filename}"
    )

    check_file_exists(filename)


# ============================================================
# STOP IF FILE MISSING
# ============================================================

missing_files = [
    filename
    for filename in PROJECT_FILES.values()
    if not Path(filename).exists()
]


if missing_files:

    print("\n❌ VALIDATION FAILED")

    print("\nMissing files:")

    for filename in missing_files:
        print(f"  - {filename}")

    print("\nNo merge was performed.")

    sys.exit(1)


# ============================================================
# READ DATASETS
# ============================================================

try:

    datasets = {}

    for dataset, filename in PROJECT_FILES.items():

        datasets[dataset] = pd.read_csv(
            filename
        )

except Exception as e:

    print(
        "\n❌ Could not read CSV files."
    )

    print(
        f"Error: {e}"
    )

    print(
        "\nNo merge was performed."
    )

    sys.exit(1)


# ============================================================
# DATASET SIZES
# ============================================================

print("\n" + "=" * 70)
print("DATASET SIZES")
print("=" * 70)


total_rows_before = 0


for dataset, df in datasets.items():

    print(
        f"{dataset.capitalize():8s}: "
        f"{len(df):,} rows × "
        f"{len(df.columns)} columns"
    )

    total_rows_before += len(df)


print(
    f"Total    : "
    f"{total_rows_before:,} rows"
)


# ============================================================
# REMOVE LANGUAGE
# ============================================================

print("\n" + "=" * 70)
print("NORMALIZING SCHEMA")
print("=" * 70)


for dataset in datasets:

    df = datasets[dataset]

    if "language" in df.columns:

        print(
            f"Dropping 'language' "
            f"from {dataset}."
        )

        datasets[dataset] = df.drop(
            columns=["language"]
        )


# ============================================================
# NORMALIZE PROJECT VALUES
# ============================================================

print("\n" + "=" * 70)
print("NORMALIZING PROJECT VALUES")
print("=" * 70)


for dataset in datasets:

    expected_project = (
        EXPECTED_PROJECTS[dataset]
    )

    print(
        f"{dataset.capitalize():8s} -> "
        f"{expected_project}"
    )

    datasets[dataset] = normalize_project(
        datasets[dataset],
        dataset,
        expected_project
    )


# ============================================================
# INDIVIDUAL DATASET VALIDATION
# ============================================================

for dataset, df in datasets.items():

    check_duplicate_columns(
        df,
        dataset
    )

    check_schema(
        df,
        dataset
    )

    check_duplicate_commit_ids(
        df,
        dataset
    )

    check_nulls(
        df,
        dataset
    )

    check_empty_strings(
        df,
        dataset
    )

    check_binary_columns(
        df,
        dataset
    )

    check_numeric_columns(
        df,
        dataset
    )

    check_date_column(
        df,
        dataset
    )

    check_row_count(
        df,
        dataset
    )


# ============================================================
# CROSS-DATASET SCHEMA CHECK
# ============================================================

print("\n" + "=" * 70)
print("CROSS-DATASET SCHEMA CHECK")
print("=" * 70)


schemas = {}

for dataset, df in datasets.items():

    schemas[dataset] = list(
        df.columns
    )

    print(
        f"{dataset}: "
        f"{len(df.columns)} columns"
    )


all_schemas_match = all(
    schema == EXPECTED_COLUMNS
    for schema in schemas.values()
)


if not all_schemas_match:

    report(
        "SCHEMA",
        "The C++ datasets do not have "
        "identical 20-column schemas "
        "after normalization."
    )


# ============================================================
# VALIDATION RESULT BEFORE TRIMMING
# ============================================================

print("\n" + "=" * 70)
print("PRE-TRIMMING VALIDATION")
print("=" * 70)


if anomalies:

    print(
        f"\n❌ {len(anomalies)} "
        f"ANOMALY/ANOMALIES FOUND"
    )

    print(
        "\nThe datasets WILL NOT be modified "
        "or merged.\n"
    )

    for i, anomaly in enumerate(
        anomalies,
        start=1
    ):

        print(
            f"{i}. {anomaly}"
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "PROCESS ABORTED"
    )

    print(
        "=" * 70
    )

    sys.exit(1)


print(
    "\n✅ PRE-TRIMMING VALIDATION PASSED"
)

print(
    "No anomalies found in the source datasets."
)


# ============================================================
# CALCULATE HOW MANY KRITA ROWS TO DROP
# ============================================================

current_total = sum(
    len(df)
    for df in datasets.values()
)


krita_count = len(
    datasets["KRITA"]
)

godot_count = len(
    datasets["GODOT"]
)

opencv_count = len(
    datasets["OPENCV"]
)


rows_to_drop = (
    current_total
    - EXPECTED_TOTAL_ROWS
)


print("\n" + "=" * 70)
print("ROW REDUCTION")
print("=" * 70)


print(
    f"Current total rows : "
    f"{current_total:,}"
)

print(
    f"Target total rows  : "
    f"{EXPECTED_TOTAL_ROWS:,}"
)

print(
    f"Rows to remove     : "
    f"{rows_to_drop:,}"
)


# ============================================================
# SAFETY CHECK:
# ALL EXTRA ROWS MUST COME FROM KRITA
# ============================================================

rows_after_removing_krita = (
    current_total - rows_to_drop
)


if (
    rows_to_drop > 0
    and rows_to_drop >= krita_count
):

    print(
        "\n❌ Cannot perform requested "
        "chronological trimming."
    )

    print(
        "There are not enough Krita rows "
        "to remove while preserving "
        "all Godot and OpenCV rows."
    )

    sys.exit(1)


# ============================================================
# VERIFY TARGET KRITA SIZE
# ============================================================

target_krita_rows = (
    EXPECTED_TOTAL_ROWS
    - godot_count
    - opencv_count
)


if target_krita_rows < 0:

    print(
        "\n❌ Target size is impossible."
    )

    print(
        f"Godot + OpenCV already contain "
        f"{godot_count + opencv_count:,} rows, "
        f"which exceeds the target of "
        f"{EXPECTED_TOTAL_ROWS:,}."
    )

    sys.exit(1)


print(
    f"\nKrita current rows : "
    f"{krita_count:,}"
)

print(
    f"Krita rows retained: "
    f"{target_krita_rows:,}"
)

print(
    f"Krita rows dropped : "
    f"{krita_count - target_krita_rows:,}"
)


# ============================================================
# CHRONOLOGICAL KRITA TRIMMING
# ============================================================

krita_df = datasets["KRITA"].copy()


# ------------------------------------------------------------
# Parse dates
# ------------------------------------------------------------

krita_dates = pd.to_datetime(
    krita_df[DATE_COLUMN],
    errors="coerce"
)


if krita_dates.isna().any():

    # This should normally already have been caught by
    # check_date_column(), but keep this safety check here.
    print(
        "\n❌ Krita date parsing failed."
    )

    print(
        "No rows were dropped and no merge "
        "was performed."
    )

    sys.exit(1)


# ------------------------------------------------------------
# Keep original row order as a tie-breaker
# ------------------------------------------------------------

krita_df["_original_order"] = range(
    len(krita_df)
)

krita_df["_parsed_date"] = krita_dates


# ------------------------------------------------------------
# Sort oldest -> newest
#
# Oldest commits will appear first.
# Newest commits will appear last.
# ------------------------------------------------------------

krita_sorted = krita_df.sort_values(
    by=[
        "_parsed_date",
        "_original_order",
    ],
    ascending=[
        True,
        True,
    ],
    kind="stable",
)


# ------------------------------------------------------------
# Drop the oldest rows
# ------------------------------------------------------------

krita_extra = krita_sorted.iloc[
    :krita_count - target_krita_rows
].copy()


# ------------------------------------------------------------
# Keep the newest rows
# ------------------------------------------------------------

krita_retained = krita_sorted.iloc[
    krita_count - target_krita_rows:
].copy()


# ------------------------------------------------------------
# Remove temporary helper columns
# ------------------------------------------------------------

krita_extra = krita_extra.drop(
    columns=[
        "_original_order",
        "_parsed_date",
    ]
)

krita_retained = krita_retained.drop(
    columns=[
        "_original_order",
        "_parsed_date",
    ]
)


# ============================================================
# VERIFY KRITA SPLIT
# ============================================================

if (
    len(krita_extra)
    + len(krita_retained)
    != krita_count
):

    print(
        "\n❌ Krita split safety check failed."
    )

    print(
        "No files were written."
    )

    sys.exit(1)


# ============================================================
# VERIFY CHRONOLOGICAL SPLIT
# ============================================================

extra_dates = pd.to_datetime(
    krita_extra[DATE_COLUMN],
    errors="coerce"
)

retained_dates = pd.to_datetime(
    krita_retained[DATE_COLUMN],
    errors="coerce"
)


if len(krita_extra) > 0:

    oldest_retained_date = (
        retained_dates.min()
    )

    newest_extra_date = (
        extra_dates.max()
    )

    if newest_extra_date > oldest_retained_date:

        print(
            "\n❌ Chronological split "
            "safety check failed."
        )

        print(
            "Some dropped rows are newer "
            "than retained rows."
        )

        print(
            "No files were written."
        )

        sys.exit(1)


# ============================================================
# STORE KRITA EXTRA ROWS IN SAME DIRECTORY
# ============================================================

krita_extra.to_csv(
    KRITA_EXTRA_FILE,
    index=False
)


print("\n" + "=" * 70)
print("KRITA ROW REMOVAL")
print("=" * 70)


print(
    f"Dropped oldest Krita rows: "
    f"{len(krita_extra):,}"
)

print(
    f"Retained newest Krita rows: "
    f"{len(krita_retained):,}"
)

print(
    f"Extra rows saved to: "
    f"{KRITA_EXTRA_FILE}"
)


# ============================================================
# UPDATE DATASETS FOR MERGE
# ============================================================

datasets["KRITA"] = krita_retained


# ============================================================
# FINAL ROW COUNT BEFORE MERGE
# ============================================================

final_row_count = sum(
    len(df)
    for df in datasets.values()
)


print("\n" + "=" * 70)
print("FINAL ROW COUNT BEFORE MERGE")
print("=" * 70)


print(
    f"Godot  : {len(datasets['GODOT']):,}"
)

print(
    f"Krita  : {len(datasets['KRITA']):,}"
)

print(
    f"OpenCV : {len(datasets['OPENCV']):,}"
)

print(
    f"Total  : {final_row_count:,}"
)


if (
    final_row_count
    != EXPECTED_TOTAL_ROWS
):

    print(
        "\n❌ FINAL ROW COUNT CHECK FAILED."
    )

    print(
        f"Expected "
        f"{EXPECTED_TOTAL_ROWS:,}, "
        f"got {final_row_count:,}."
    )

    print(
        "The final merged dataset "
        "will NOT be written."
    )

    sys.exit(1)


# ============================================================
# FINAL PRE-MERGE SCHEMA CHECK
# ============================================================

for dataset, df in datasets.items():

    if list(df.columns) != EXPECTED_COLUMNS:

        print(
            f"\n❌ FINAL SCHEMA CHECK FAILED "
            f"FOR {dataset}."
        )

        print(
            "The final merged dataset "
            "will NOT be written."
        )

        sys.exit(1)


# ============================================================
# MERGE
# ============================================================

print("\n" + "=" * 70)
print("MERGING DATASETS")
print("=" * 70)


final_df = pd.concat(
    [
        datasets["GODOT"],
        datasets["KRITA"],
        datasets["OPENCV"],
    ],
    ignore_index=True
)


# ============================================================
# FINAL SAFETY CHECKS
# ============================================================

print(
    "\nRunning final safety checks..."
)


# ------------------------------------------------------------
# Row count
# ------------------------------------------------------------

if len(final_df) != EXPECTED_TOTAL_ROWS:

    print(
        "\n❌ SAFETY CHECK FAILED: "
        "incorrect final row count."
    )

    print(
        "Final file will NOT be written."
    )

    sys.exit(1)


# ------------------------------------------------------------
# Column count
# ------------------------------------------------------------

if len(final_df.columns) != 20:

    print(
        "\n❌ SAFETY CHECK FAILED: "
        "final dataset does not contain "
        "exactly 20 columns."
    )

    print(
        "Final file will NOT be written."
    )

    sys.exit(1)


# ------------------------------------------------------------
# Exact schema
# ------------------------------------------------------------

if list(final_df.columns) != EXPECTED_COLUMNS:

    print(
        "\n❌ SAFETY CHECK FAILED: "
        "final schema is incorrect."
    )

    print(
        "Final file will NOT be written."
    )

    sys.exit(1)


# ------------------------------------------------------------
# Final project values
# ------------------------------------------------------------

expected_final_projects = {
    "cpp/godot",
    "cpp/krita",
    "cpp/opencv",
}


actual_final_projects = set(
    final_df["project"]
    .dropna()
    .astype(str)
    .str.strip()
    .str.lower()
    .unique()
)


if (
    actual_final_projects
    != expected_final_projects
):

    print(
        "\n❌ SAFETY CHECK FAILED: "
        "unexpected project values."
    )

    print(
        f"Expected: "
        f"{sorted(expected_final_projects)}"
    )

    print(
        f"Found:    "
        f"{sorted(actual_final_projects)}"
    )

    print(
        "Final file will NOT be written."
    )

    sys.exit(1)


# ------------------------------------------------------------
# Final duplicate commit check
# ------------------------------------------------------------

duplicate_final_ids = (
    final_df["commit_id"]
    .duplicated(keep=False)
)


if duplicate_final_ids.any():

    duplicate_ids = (
        final_df.loc[
            duplicate_final_ids,
            "commit_id"
        ]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    print(
        "\n❌ SAFETY CHECK FAILED: "
        "duplicate commit IDs in "
        "final dataset."
    )

    print(
        f"Number of duplicate IDs: "
        f"{len(duplicate_ids)}"
    )

    print(
        f"Examples: "
        f"{duplicate_ids[:10]}"
    )

    print(
        "Final file will NOT be written."
    )

    sys.exit(1)


# ============================================================
# SAVE FINAL DATASET
# ============================================================

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUCCESS REPORT
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "SUCCESS"
)

print(
    "=" * 70
)


print(
    f"\nFinal dataset written to:"
)

print(
    f"  {OUTPUT_FILE}"
)


print(
    f"\nFinal shape:"
)

print(
    f"  Rows    : "
    f"{len(final_df):,}"
)

print(
    f"  Columns : "
    f"{len(final_df.columns)}"
)


print(
    "\nFinal project distribution:"
)

print(
    final_df["project"]
    .value_counts()
    .to_string()
)


print(
    "\nFinal project values:"
)

for value in sorted(
    final_df["project"].unique()
):

    print(
        f"  - {value}"
    )


print(
    "\nKrita extra rows:"
)

print(
    f"  {KRITA_EXTRA_FILE}"
)

print(
    f"  Rows removed: "
    f"{len(krita_extra):,}"
)


print(
    "\n✅ C++ 12K dataset is ready."
)