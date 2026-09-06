import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


csv.field_size_limit(2_147_483_647)

JAVA_PATH = Path(r"D:\embeddings_and_pca_code\data\java_only_latest_36k_pca384.csv")
PYCPP_PATH = Path(r"D:\embeddings_and_pca_code\data\python_cpp_23996_pca384.csv")
OUTPUT_PATH = Path(r"D:\embeddings_and_pca_code\data\final_multilanguage_59996_pca384.csv")

PYCPP_COLUMNS = [
    "commit_id", "project", "buggy", "fix", "year", "author_date",
    "la", "ld", "nf", "nd", "ns", "ent", "ndev", "age", "nuc",
    "aexp", "arexp", "asexp", "message", "diff", "embedding",
]
JAVA_COLUMNS = PYCPP_COLUMNS[:-1] + ["input_text", "embedding"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_date(value):
    if value.isdigit():
        return datetime.fromtimestamp(int(value), timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def parse_buggy(value):
    labels = {"0": 0, "1": 1, "False": 0, "True": 1, "false": 0, "true": 1}
    if value not in labels:
        raise ValueError(f"Unexpected buggy value: {value}")
    return str(labels[value])


def inspect_file(path, expected_rows, expected_family, expected_columns):
    if not path.exists():
        raise ValueError(f"Missing input: {path}")

    rows = 0
    columns = None
    commit_ids = set()
    projects = {}
    buggy_counts = {"0": 0, "1": 0}
    missing_cells = 0
    malformed_embeddings = 0
    nonfinite_values = 0
    embedding_count = 0
    embedding_min = math.inf
    embedding_max = -math.inf
    embedding_sum = 0.0
    embedding_sum_sq = 0.0
    date_values = []
    input_text_nonnull = 0
    input_text_empty = 0
    input_text_missing = 0

    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames
        if columns != expected_columns:
            raise ValueError(f"Schema mismatch in {path}: {columns}")

        for row in reader:
            rows += 1
            commit_id = row["commit_id"]
            if not commit_id:
                raise ValueError(f"Missing commit_id in {path}, row {rows}")
            if commit_id in commit_ids:
                raise ValueError(f"Duplicate commit_id in {path}: {commit_id}")
            commit_ids.add(commit_id)

            project = row["project"]
            if not project.startswith(expected_family):
                raise ValueError(f"Unexpected project family in {path}: {project}")
            projects[project] = projects.get(project, 0) + 1

            for column in expected_columns:
                if column == "input_text":
                    continue
                if not row[column]:
                    missing_cells += 1

            if "input_text" in expected_columns:
                if row["input_text"] is None:
                    input_text_missing += 1
                elif row["input_text"] == "":
                    input_text_empty += 1
                else:
                    input_text_nonnull += 1

            try:
                buggy_counts[parse_buggy(row["buggy"])] += 1
            except ValueError as error:
                raise ValueError(f"Unexpected buggy value in {path}: {row['buggy']}") from error

            try:
                date_values.append(parse_date(row["author_date"]))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid author_date in {path}: {row['author_date']}") from error

            try:
                embedding = json.loads(row["embedding"])
                if not isinstance(embedding, list) or len(embedding) != 384:
                    raise ValueError("wrong dimension")
                for value in embedding:
                    number = float(value)
                    if not math.isfinite(number):
                        nonfinite_values += 1
                    embedding_min = min(embedding_min, number)
                    embedding_max = max(embedding_max, number)
                    embedding_sum += number
                    embedding_sum_sq += number * number
                    embedding_count += 1
            except (TypeError, ValueError, json.JSONDecodeError):
                malformed_embeddings += 1

    if rows != expected_rows:
        raise ValueError(f"Row count mismatch in {path}: {rows}, expected {expected_rows}")
    if malformed_embeddings:
        raise ValueError(f"Malformed embeddings in {path}: {malformed_embeddings}")
    if missing_cells:
        raise ValueError(f"Missing cells in {path}: {missing_cells}")

    mean = embedding_sum / embedding_count
    variance = max(0.0, embedding_sum_sq / embedding_count - mean * mean)
    return {
        "rows": rows,
        "columns": columns,
        "input_text_nonnull": input_text_nonnull,
        "input_text_empty": input_text_empty,
        "input_text_missing": input_text_missing,
        "commit_ids": commit_ids,
        "projects": projects,
        "buggy": buggy_counts,
        "missing_cells": missing_cells,
        "malformed_embeddings": malformed_embeddings,
        "nonfinite_values": nonfinite_values,
        "embedding_min": embedding_min,
        "embedding_max": embedding_max,
        "embedding_mean": mean,
        "embedding_std": math.sqrt(variance),
        "date_min": min(date_values),
        "date_max": max(date_values),
    }


def row_matches(expected, actual, ignored_columns=()):
    expected_columns = [column for column in expected if column not in ignored_columns]
    actual_columns = [column for column in actual if column not in ignored_columns]
    if expected_columns != actual_columns:
        return False
    for column in expected_columns:
        if column == "embedding":
            try:
                left = json.loads(expected[column])
                right = json.loads(actual[column])
            except (TypeError, ValueError, json.JSONDecodeError):
                return False
            if len(left) != len(right) or any(float(a) != float(b) for a, b in zip(left, right)):
                return False
        elif expected[column] != actual[column]:
            return False
    return True


def validate_section(path, start, expected_rows, ignored_columns=(), require_end=False):
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        for _ in range(start):
            if next(reader, None) is None:
                return False
        for index, expected in enumerate(expected_rows):
            actual = next(reader, None)
            if actual is None or not row_matches(expected, actual, ignored_columns):
                return False
        return not require_end or next(reader, None) is None


def main():
    if OUTPUT_PATH.exists():
        raise ValueError(f"Refusing to overwrite existing output: {OUTPUT_PATH}")

    java_hash_before = sha256(JAVA_PATH)
    pycpp_hash_before = sha256(PYCPP_PATH)
    java = inspect_file(JAVA_PATH, 36000, "apache/", JAVA_COLUMNS)
    pycpp = inspect_file(PYCPP_PATH, 23996, ("python/", "cpp/"), PYCPP_COLUMNS)

    overlap = java["commit_ids"] & pycpp["commit_ids"]
    if overlap:
        raise ValueError(f"Cross-dataset commit overlap: {sorted(overlap)[:10]}")

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as output:
        output_writer = csv.DictWriter(output, fieldnames=PYCPP_COLUMNS, lineterminator="\n")
        output_writer.writeheader()
        with JAVA_PATH.open("r", encoding="utf-8", newline="") as java_source:
            for row in csv.DictReader(java_source):
                row.pop("input_text")
                output_writer.writerow(row)
        with PYCPP_PATH.open("r", encoding="utf-8", newline="") as pycpp_source:
            for row in csv.DictReader(pycpp_source):
                output_writer.writerow(row)

    with JAVA_PATH.open("r", encoding="utf-8", newline="") as source:
        java_rows = list(csv.DictReader(source))
    with PYCPP_PATH.open("r", encoding="utf-8", newline="") as source:
        pycpp_rows = list(csv.DictReader(source))
    if not validate_section(OUTPUT_PATH, 0, java_rows, ignored_columns=("input_text",)):
        raise ValueError("Output Java rows are not preserved after input_text removal")
    if not validate_section(OUTPUT_PATH, len(java_rows), pycpp_rows, require_end=True):
        raise ValueError("Output rows are not an exact Java-first concatenation")

    output_info = inspect_file(OUTPUT_PATH, 59996, ("apache/", "python/", "cpp/"), PYCPP_COLUMNS)
    java_hash_after = sha256(JAVA_PATH)
    pycpp_hash_after = sha256(PYCPP_PATH)
    expected_buggy = {key: java["buggy"].get(key, 0) + pycpp["buggy"].get(key, 0) for key in ("0", "1")}
    expected_projects = java["projects"].copy()
    for project, count in pycpp["projects"].items():
        expected_projects[project] = expected_projects.get(project, 0) + count

    checks = [
        output_info["rows"] == 59996,
        output_info["columns"] == PYCPP_COLUMNS,
        output_info["buggy"] == expected_buggy,
        output_info["projects"] == expected_projects,
        output_info["nonfinite_values"] == 0,
        len(output_info["commit_ids"]) == 59996,
        not overlap,
        java_hash_before == java_hash_after,
        pycpp_hash_before == pycpp_hash_after,
    ]

    print("PRE-MERGE VALIDATION")
    print(f"Java rows: {java['rows']}")
    print(f"Python/C++ rows: {pycpp['rows']}")
    print("Expected final rows: 59996")
    print(f"Schema comparison: Python/C++ base schema valid; Java has only input_text extra: {java['columns'] == JAVA_COLUMNS and pycpp['columns'] == PYCPP_COLUMNS}")
    print(f"Java input_text: non-null={java['input_text_nonnull']}, empty={java['input_text_empty']}, missing={java['input_text_missing']}")
    print(f"Embedding dimensions: Java 384D; Python/C++ 384D")
    print(f"Missing/non-finite embeddings: Java {java['missing_cells']}/{java['nonfinite_values']}; Python/C++ {pycpp['missing_cells']}/{pycpp['nonfinite_values']}")
    print(f"Duplicate commit IDs: Java 0; Python/C++ 0")
    print(f"Cross-dataset overlap count: {len(overlap)}")
    print(f"Buggy counts: Java {java['buggy']}; Python/C++ {pycpp['buggy']}")
    print(f"Java projects: {java['projects']}")
    print(f"Python/C++ projects: {pycpp['projects']}")
    print(f"Date ranges: Java {java['date_min']} to {java['date_max']}; Python/C++ {pycpp['date_min']} to {pycpp['date_max']}")
    print(f"Java embedding stats: min={java['embedding_min']}, max={java['embedding_max']}, mean={java['embedding_mean']}, std={java['embedding_std']}")
    print(f"Python/C++ embedding stats: min={pycpp['embedding_min']}, max={pycpp['embedding_max']}, mean={pycpp['embedding_mean']}, std={pycpp['embedding_std']}")
    print(f"Java SHA-256: {java_hash_before}")
    print(f"Python/C++ SHA-256: {pycpp_hash_before}")
    print("MERGE")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Final row count: {output_info['rows']}")
    print("Row ordering: Java 36,000 first, Python/C++ 23,996 second")
    print("POST-MERGE VALIDATION")
    print(f"Final row count: {output_info['rows']}")
    print(f"Schema: {output_info['columns'] == PYCPP_COLUMNS}")
    print("Duplicate IDs: 0")
    print(f"Embedding validation: {output_info['nonfinite_values'] == 0 and output_info['missing_cells'] == 0}")
    print("Row preservation: exact Java-first concatenation")
    print(f"Input hash preservation: {java_hash_before == java_hash_after and pycpp_hash_before == pycpp_hash_after}")
    print(f"FINAL MERGE VALIDATION: {'PASS' if all(checks) else 'FAIL'}")


if __name__ == "__main__":
    main()