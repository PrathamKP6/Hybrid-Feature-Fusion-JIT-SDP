import csv
import sys

csv.field_size_limit(2_000_000_000)
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = Path(r"D:\embeddings_and_pca_code\data\final_multilanguage_59996_pca384.csv")

OUTPUT_FILE = BASE_DIR / "data" / "llm_input_59996.csv"


# ============================================================
# POSSIBLE COLUMN NAMES
# The script will detect the actual names automatically.
# ============================================================

COLUMN_CANDIDATES = {
    "commit_id": [
        "commit_id",
        "commitid",
        "commit_hash",
        "commithash",
        "hash",
        "sha",
        "commit_sha",
        "commit"
    ],

    "commit_message": [
        "commit_message",
        "commitmessage",
        "message",
        "commit_msg",
        "commitmsg",
        "msg"
    ],

    "commit_diff": [
        "commit_diff",
        "commitdiff",
        "diff",
        "code_diff",
        "codediff",
        "patch",
        "commit_patch"
    ]
}


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

def normalize(name):
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def find_column(headers, candidates, field_name):

    normalized_headers = {
        normalize(h): h
        for h in headers
    }

    # Exact normalized match first
    for candidate in candidates:
        if candidate in normalized_headers:
            return normalized_headers[candidate]

    # Partial match second
    possible = []

    for header in headers:
        h = normalize(header)

        for candidate in candidates:
            if candidate in h or h in candidate:
                possible.append(header)
                break

    possible = list(dict.fromkeys(possible))

    if len(possible) == 1:
        return possible[0]

    print(f"\nERROR: Could not uniquely identify {field_name}.")
    print("Possible columns:")
    for col in possible:
        print("   ", col)

    print("\nAll columns in the dataset:")
    for col in headers:
        print("   ", col)

    raise RuntimeError(
        f"Please identify the correct column for {field_name}."
    )


# ============================================================
# CHECK INPUT
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}\n"
        "\nMake sure final_multilanguage_59996_pca384.csv "
        "is inside the 'data' folder."
    )


print("=" * 70)
print("LLM DATASET PREPARATION")
print("=" * 70)

print(f"\nInput : {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")


# ============================================================
# READ HEADER ONLY FIRST
# ============================================================

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8",
    errors="replace",
    newline=""
) as f:

    reader = csv.reader(f)

    headers = next(reader)

print("\nDetected dataset columns:")
for h in headers:
    print("   ", h)


# ============================================================
# FIND REQUIRED COLUMNS
# ============================================================

commit_id_col = find_column(
    headers,
    COLUMN_CANDIDATES["commit_id"],
    "commit_id"
)

message_col = find_column(
    headers,
    COLUMN_CANDIDATES["commit_message"],
    "commit_message"
)

diff_col = find_column(
    headers,
    COLUMN_CANDIDATES["commit_diff"],
    "commit_diff"
)


print("\n" + "=" * 70)
print("COLUMN MAPPING")
print("=" * 70)

print(f"commit_id      -> {commit_id_col}")
print(f"commit_message -> {message_col}")
print(f"commit_diff    -> {diff_col}")


# ============================================================
# CREATE SMALL LLM INPUT FILE
# ============================================================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

rows_written = 0
empty_message = 0
empty_diff = 0

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8",
    errors="replace",
    newline=""
) as infile, open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
    newline=""
) as outfile:

    reader = csv.DictReader(infile)

    writer = csv.DictWriter(
        outfile,
        fieldnames=[
            "commit_id",
            "commit_message",
            "commit_diff"
        ]
    )

    writer.writeheader()

    for row in reader:

        commit_id = row.get(commit_id_col, "")
        message = row.get(message_col, "")
        diff = row.get(diff_col, "")

        if not message:
            empty_message += 1

        if not diff:
            empty_diff += 1

        writer.writerow({
            "commit_id": commit_id,
            "commit_message": message,
            "commit_diff": diff
        })

        rows_written += 1

        if rows_written % 5000 == 0:
            print(f"Processed {rows_written:,} rows...")


# ============================================================
# FINAL REPORT
# ============================================================

output_size_mb = OUTPUT_FILE.stat().st_size / (1024 ** 2)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(f"Rows written       : {rows_written:,}")
print(f"Empty messages     : {empty_message:,}")
print(f"Empty diffs        : {empty_diff:,}")
print(f"Output size        : {output_size_mb:.2f} MB")
print(f"Output file        : {OUTPUT_FILE}")

print("\nColumns:")
print("  1. commit_id")
print("  2. commit_message")
print("  3. commit_diff")

print("\nReady for GPU LLM feature extraction.")