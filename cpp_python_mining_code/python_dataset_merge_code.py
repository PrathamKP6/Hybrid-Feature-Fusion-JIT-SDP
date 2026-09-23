import pandas as pd
from pathlib import Path
import sys

# ============================================================
# CONFIG
# ============================================================

DJANGO_FILE = "python_django_dataset.csv"
FLASK_FILE = "python_flask_dataset.csv"
CPYTHON_FILE = "python_cpython_dataset.csv"

OUTPUT_FILE = "final_python_12k.csv"

EXPECTED_TOTAL_ROWS = 12000

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

# Exact project values wanted in the final dataset
EXPECTED_PROJECTS = {
    "DJANGO": "python/django",
    "FLASK": "python/flask",
    "CPYTHON": "python/cpython",
}

# Columns that should not normally be empty
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
    anomalies.append(f"[{dataset}] {message}")


# ============================================================
# FILE CHECK
# ============================================================

def check_file_exists(path):
    if not Path(path).exists():
        report("FILES", f"Missing file: {path}")
        return False
    return True


# ============================================================
# COLUMN CHECKS
# ============================================================

def check_duplicate_columns(df, dataset):
    duplicated = df.columns[df.columns.duplicated()].tolist()

    if duplicated:
        report(
            dataset,
            f"Duplicate column names found: {duplicated}"
        )


def check_schema(df, dataset):
    actual = list(df.columns)

    missing = [
        c for c in EXPECTED_COLUMNS
        if c not in actual
    ]

    extra = [
        c for c in actual
        if c not in EXPECTED_COLUMNS
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

    # Check order only when columns themselves are correct
    if not missing and not extra:
        if actual != EXPECTED_COLUMNS:
            report(
                dataset,
                "Column order is incorrect.\n"
                f"Expected: {EXPECTED_COLUMNS}\n"
                f"Found:    {actual}"
            )


# ============================================================
# PROJECT CHECK
# ============================================================

def normalize_project(df, dataset, expected_project):
    """
    Force the project column to the required canonical value.

    Django  -> python/django
    Flask   -> python/flask
    CPython -> python/cpython

    Before changing it, report any unexpected existing values.
    """

    if "project" not in df.columns:
        report(
            dataset,
            "Missing 'project' column."
        )
        return df

    # Existing values before normalization
    existing_values = set(
        df["project"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )

    # Empty / null project values are still anomalies
    null_count = df["project"].isna().sum()

    if null_count > 0:
        report(
            dataset,
            f"'project' contains {null_count} null values."
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
            f"'project' contains {empty_count} empty values."
        )

    # We accept either the already-correct value or the
    # short project name produced by the mining scripts.
    #
    # For example:
    # django -> python/django
    # python/django -> python/django
    accepted_values = {
        expected_project,
        expected_project.replace("python/", "")
    }

    unexpected = existing_values - accepted_values

    if unexpected:
        report(
            dataset,
            f"Unexpected project values: "
            f"{sorted(unexpected)}. "
            f"Expected values equivalent to: "
            f"{expected_project}"
        )
        return df

    # Normalize everything to the required final value
    df["project"] = expected_project

    return df


# ============================================================
# DUPLICATE COMMIT CHECK
# ============================================================

def check_duplicate_commit_ids(df, dataset):
    if "commit_id" not in df.columns:
        return

    duplicates = df[
        df["commit_id"].duplicated(keep=False)
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

def check_empty_strings(df, dataset):

    for col in [
        "commit_id",
        "project",
        "author_date"
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
# BUG/FIX BINARY CHECK
# ============================================================

def check_binary_columns(df, dataset):

    for col in ["buggy", "fix"]:

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
            "true"
        }

        invalid = values - allowed

        if invalid:
            report(
                dataset,
                f"Column '{col}' contains "
                f"unexpected values: {sorted(invalid)}"
            )


# ============================================================
# NUMERIC COLUMN CHECK
# ============================================================

def check_numeric_columns(df, dataset):

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

        if invalid.sum() > 0:
            report(
                dataset,
                f"Column '{col}' contains "
                f"{invalid.sum()} non-numeric values."
            )


# ============================================================
# ROW COUNT CHECK
# ============================================================

def check_row_count(df, dataset):

    if len(df) == 0:
        report(
            dataset,
            "Dataset contains 0 rows."
        )


# ============================================================
# LOAD FILES
# ============================================================

print("=" * 70)
print("FINAL PYTHON DATASET VALIDATION")
print("=" * 70)

files = {
    "DJANGO": DJANGO_FILE,
    "FLASK": FLASK_FILE,
    "CPYTHON": CPYTHON_FILE,
}

for name, path in files.items():

    print(f"\nChecking file: {path}")

    check_file_exists(path)


# ============================================================
# STOP IF FILE MISSING
# ============================================================

if any(
    not Path(path).exists()
    for path in files.values()
):

    print("\n❌ VALIDATION FAILED")
    print("\nMissing files:")

    for dataset, path in files.items():

        if not Path(path).exists():
            print(f"  - {path}")

    print("\nNo merge was performed.")

    sys.exit(1)


# ============================================================
# READ CSV FILES
# ============================================================

try:

    django = pd.read_csv(DJANGO_FILE)
    flask = pd.read_csv(FLASK_FILE)
    cpython = pd.read_csv(CPYTHON_FILE)

except Exception as e:

    print("\n❌ Could not read CSV files.")
    print(f"Error: {e}")
    print("\nNo merge was performed.")

    sys.exit(1)


# ============================================================
# DATASET SIZE
# ============================================================

print("\n" + "=" * 70)
print("DATASET SIZES")
print("=" * 70)

print(
    f"Django :  {len(django):,} rows × "
    f"{len(django.columns)} columns"
)

print(
    f"Flask  :  {len(flask):,} rows × "
    f"{len(flask.columns)} columns"
)

print(
    f"CPython:  {len(cpython):,} rows × "
    f"{len(cpython.columns)} columns"
)

print(
    f"Total   : "
    f"{len(django) + len(flask) + len(cpython):,} rows"
)


# ============================================================
# NORMALIZE LANGUAGE COLUMN
# ============================================================

print("\n" + "=" * 70)
print("NORMALIZING SCHEMA")
print("=" * 70)

if "language" in django.columns:

    print("Dropping 'language' from Django.")

    django = django.drop(
        columns=["language"]
    )


if "language" in flask.columns:

    print("Dropping 'language' from Flask.")

    flask = flask.drop(
        columns=["language"]
    )


if "language" in cpython.columns:

    report(
        "CPYTHON",
        "Unexpected 'language' column found."
    )


# ============================================================
# NORMALIZE PROJECT VALUES
# ============================================================

print("\n" + "=" * 70)
print("NORMALIZING PROJECT VALUES")
print("=" * 70)

django = normalize_project(
    django,
    "DJANGO",
    EXPECTED_PROJECTS["DJANGO"]
)

flask = normalize_project(
    flask,
    "FLASK",
    EXPECTED_PROJECTS["FLASK"]
)

cpython = normalize_project(
    cpython,
    "CPYTHON",
    EXPECTED_PROJECTS["CPYTHON"]
)

print(
    "Django  -> python/django"
)

print(
    "Flask   -> python/flask"
)

print(
    "CPython -> python/cpython"
)


# ============================================================
# RUN DATASET VALIDATIONS
# ============================================================

datasets = {
    "DJANGO": django,
    "FLASK": flask,
    "CPYTHON": cpython,
}

for dataset_name, df in datasets.items():

    check_duplicate_columns(
        df,
        dataset_name
    )

    check_schema(
        df,
        dataset_name
    )

    check_duplicate_commit_ids(
        df,
        dataset_name
    )

    check_nulls(
        df,
        dataset_name
    )

    check_empty_strings(
        df,
        dataset_name
    )

    check_binary_columns(
        df,
        dataset_name
    )

    check_numeric_columns(
        df,
        dataset_name
    )

    check_row_count(
        df,
        dataset_name
    )


# ============================================================
# CROSS-DATASET SCHEMA CHECK
# ============================================================

print("\n" + "=" * 70)
print("CROSS-DATASET SCHEMA CHECK")
print("=" * 70)

schemas = {
    "DJANGO": list(django.columns),
    "FLASK": list(flask.columns),
    "CPYTHON": list(cpython.columns),
}

for name, schema in schemas.items():

    print(
        f"{name}: {len(schema)} columns"
    )

if not (
    schemas["DJANGO"]
    == schemas["FLASK"]
    == schemas["CPYTHON"]
    == EXPECTED_COLUMNS
):

    report(
        "SCHEMA",
        "The three datasets do not have "
        "identical 20-column schemas."
    )


# ============================================================
# TOTAL ROW COUNT CHECK
# ============================================================

total_rows = (
    len(django)
    + len(flask)
    + len(cpython)
)

print("\n" + "=" * 70)
print("TOTAL ROW CHECK")
print("=" * 70)

print(
    f"Expected total rows: "
    f"{EXPECTED_TOTAL_ROWS:,}"
)

print(
    f"Actual total rows  : "
    f"{total_rows:,}"
)

if total_rows != EXPECTED_TOTAL_ROWS:

    report(
        "TOTAL",
        f"Expected {EXPECTED_TOTAL_ROWS:,} rows, "
        f"but found {total_rows:,}."
    )


# ============================================================
# FINAL VALIDATION DECISION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION RESULT")
print("=" * 70)

if anomalies:

    print(
        f"\n❌ {len(anomalies)} "
        f"ANOMALY/ANOMALIES FOUND"
    )

    print(
        "\nThe datasets WILL NOT be merged.\n"
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
        "MERGE ABORTED"
    )

    print(
        "=" * 70
    )

    print(
        "\nFix the reported anomalies "
        "and run this script again."
    )

    sys.exit(1)


# ============================================================
# MERGE
# ============================================================

print(
    "\n✅ NO ANOMALIES FOUND"
)

print(
    "All validation checks passed."
)

print(
    "\nMerging datasets..."
)

final_df = pd.concat(
    [
        django,
        flask,
        cpython,
    ],
    ignore_index=True
)


# ============================================================
# POST-MERGE SAFETY CHECK
# ============================================================

if len(final_df) != EXPECTED_TOTAL_ROWS:

    print(
        "\n❌ SAFETY CHECK FAILED "
        "AFTER MERGE."
    )

    print(
        f"Expected {EXPECTED_TOTAL_ROWS:,} rows, "
        f"got {len(final_df):,}."
    )

    print(
        "The final file will NOT be written."
    )

    sys.exit(1)


if list(final_df.columns) != EXPECTED_COLUMNS:

    print(
        "\n❌ SAFETY CHECK FAILED: "
        "final schema is incorrect."
    )

    print(
        "The final file will NOT be written."
    )

    sys.exit(1)


# ============================================================
# FINAL PROJECT VALUE CHECK
# ============================================================

expected_final_projects = {
    "python/django",
    "python/flask",
    "python/cpython",
}

actual_final_projects = set(
    final_df["project"]
    .dropna()
    .astype(str)
    .str.strip()
    .str.lower()
    .unique()
)

if actual_final_projects != expected_final_projects:

    print(
        "\n❌ SAFETY CHECK FAILED: "
        "unexpected final project values."
    )

    print(
        f"Expected: {sorted(expected_final_projects)}"
    )

    print(
        f"Found:    {sorted(actual_final_projects)}"
    )

    print(
        "The final file will NOT be written."
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
    f"  Rows    : {len(final_df):,}"
)

print(
    f"  Columns : {len(final_df.columns)}"
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
    "\nFinal columns:"
)

for i, col in enumerate(
    final_df.columns,
    start=1
):

    print(
        f"  {i:2d}. {col}"
    )

print(
    "\n✅ final_python_12k.csv is ready."
)