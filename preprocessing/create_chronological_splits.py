"""Create project-wise chronological JIT-SDP splits without loading embeddings.

The first pass loads only metadata columns. The second pass reads complete CSV
rows in chunks and copies their original fields through temporary buckets. Legitimate
historical dates (including pre-2000 CPython commits) are retained. No PCA, scaling,
normalization, sampling, or randomization is performed. Output headers, commit
coverage, and raw embedding strings are validated after writing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "final_multilanguage_59996_szz_buggy.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "chronological_splits"
REQUIRED_COLUMNS = ("commit_id", "project", "author_date", "buggy")
LANGUAGE_PREFIXES = {"apache/": "java", "cpp/": "cpp", "python/": "python"}
LANGUAGES = ("java", "cpp", "python")
DEFAULT_CHUNK_SIZE = 500
SANITY_LOWER_BOUND = pd.Timestamp("1980-01-01", tz="UTC")
SANITY_UPPER_BOUND = pd.Timestamp(datetime.now(timezone.utc).date(), tz="UTC") + pd.Timedelta(days=366)


def _read_header(path: Path) -> list[str]:
    header = pd.read_csv(path, nrows=0)
    missing = [column for column in REQUIRED_COLUMNS if column not in header.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return list(header.columns)


def derive_language(projects: pd.Series) -> pd.Series:
    projects = projects.astype("string")
    language = pd.Series(pd.NA, index=projects.index, dtype="string")
    for prefix, value in LANGUAGE_PREFIXES.items():
        language.loc[projects.str.startswith(prefix, na=False)] = value
    if language.isna().any():
        examples = projects.loc[language.isna()].drop_duplicates().head(10).tolist()
        raise ValueError(f"Unknown project prefix in project values: {examples}")
    return language.astype(str)


def _valid_date(value: pd.Timestamp) -> bool:
    return pd.notna(value) and SANITY_LOWER_BOUND <= value <= SANITY_UPPER_BOUND


def _parse_one_date(raw: str) -> tuple[pd.Timestamp, str]:
    """Parse one value by testing supported units against sanity bounds."""
    text = str(raw).strip()
    numeric = pd.to_numeric(text, errors="coerce")
    if pd.notna(numeric):
        candidates = []
        for unit in ("s", "ms", "us", "ns"):
            candidate = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
            if _valid_date(candidate):
                candidates.append((candidate, f"unix_{unit}"))
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError(f"Numeric author_date {text!r} has no valid seconds/ms/us/ns interpretation")
        raise ValueError(f"Numeric author_date {text!r} has multiple valid unit interpretations")

    candidate = pd.to_datetime(text, format="mixed", utc=True, errors="coerce")
    if _valid_date(candidate):
        return candidate, "datetime_string"

    # Some source rows contain a nanosecond-formatted rendering of a Unix
    # seconds value, e.g. 1970-01-01 00:00:01.577381993 for 1577381993.
    if pd.notna(candidate) and candidate < SANITY_LOWER_BOUND:
        epoch_nanoseconds = candidate.value
        repaired = pd.to_datetime(epoch_nanoseconds, unit="s", utc=True, errors="coerce")
        if _valid_date(repaired):
            return repaired, "legacy_epoch_string_as_seconds"
    raise ValueError(f"author_date {text!r} is invalid or outside {SANITY_LOWER_BOUND.date()} to {SANITY_UPPER_BOUND.date()}")


def parse_author_dates(values: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    parsed: list[pd.Timestamp] = []
    methods: dict[str, int] = {}
    invalid_values: list[str] = []
    examples = values.drop_duplicates().head(10).tolist()
    print(f"Raw author_date examples: {examples}")
    for raw in values:
        try:
            value, method = _parse_one_date(raw)
        except ValueError:
            invalid_values.append(str(raw))
            continue
        parsed.append(value)
        methods[method] = methods.get(method, 0) + 1
    if invalid_values:
        examples = invalid_values[:10]
        raise ValueError(f"Found {len(invalid_values)} invalid author_date values; examples: {examples}")
    result = pd.Series(parsed, index=values.index, dtype="datetime64[ns, UTC]")
    if result.isna().any():
        raise ValueError(f"Could not parse {int(result.isna().sum())} author_date values")
    if result.min() < SANITY_LOWER_BOUND or result.max() > SANITY_UPPER_BOUND:
        raise ValueError("Parsed author_date values violate the configured sanity bounds")
    print(f"Detected author_date representations: {methods}")
    print(f"Minimum parsed date: {result.min().isoformat()}")
    print(f"Maximum parsed date: {result.max().isoformat()}")
    return result, methods


def load_metadata(path: Path, chunk_size: int) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """Read and validate only commit_id, project, author_date, and buggy."""
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    original_columns = _read_header(path)
    parts: list[pd.DataFrame] = []
    source_row_id = 0
    for chunk in pd.read_csv(path, usecols=list(REQUIRED_COLUMNS), dtype="string", keep_default_na=False, chunksize=chunk_size):
        chunk = chunk.reset_index(drop=True)
        chunk["_source_row_id"] = range(source_row_id, source_row_id + len(chunk))
        source_row_id += len(chunk)
        parts.append(chunk)
    if not parts:
        raise ValueError("Input CSV contains no observations")
    metadata = pd.concat(parts, ignore_index=True)
    if (metadata["commit_id"] == "").any():
        raise ValueError("commit_id contains missing values")
    if (metadata["project"] == "").any():
        raise ValueError("project contains missing values")
    buggy_text = metadata["buggy"].astype("string").str.strip().str.lower().replace({"true": "1", "false": "0"})
    buggy = pd.to_numeric(buggy_text, errors="coerce")
    if buggy.isna().any() or not buggy.isin([0, 1]).all():
        invalid = metadata.loc[buggy.isna() | ~buggy.isin([0, 1]), "buggy"].value_counts(dropna=False).head(10).to_dict()
        raise ValueError(f"buggy must contain only binary values 0 and 1; invalid values: {invalid}")
    metadata["buggy"] = buggy.astype(int)
    metadata["language"] = derive_language(metadata["project"])
    metadata["_parsed_author_date"], methods = parse_author_dates(metadata["author_date"])

    pre_2000 = int((metadata["_parsed_author_date"] < pd.Timestamp("2000-01-01", tz="UTC")).sum())
    print(f"Legitimate pre-2000 dates retained: {pre_2000}")
    print(
        "Global parsed date range: "
        f"{metadata['_parsed_author_date'].min().isoformat()} -> "
        f"{metadata['_parsed_author_date'].max().isoformat()}"
    )
    return metadata, original_columns, methods


def audit_duplicates(metadata: pd.DataFrame, policy: str) -> tuple[pd.DataFrame, int, int]:
    duplicate_mask = metadata["commit_id"].duplicated(keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    duplicate_ids = int(metadata.loc[duplicate_mask, "commit_id"].nunique())
    print(f"Duplicate commit IDs: {duplicate_ids}; affected rows: {duplicate_rows}")
    if duplicate_rows and policy == "error":
        raise ValueError("Duplicate commit IDs found and duplicate policy is 'error'")
    if duplicate_rows:
        metadata = metadata.drop_duplicates("commit_id", keep="first").copy()
        print(f"Rows remaining after duplicate removal: {len(metadata)}")
    return metadata, duplicate_ids, duplicate_rows


def _allocation(total: int) -> tuple[int, int]:
    """Return deterministic chronological boundaries, keeping 3-way splits nonempty when possible."""
    if total < 3:
        return (1 if total >= 1 else 0, total if total == 2 else 1 if total == 1 else 0)
    train_end = round(total * 0.70)
    validation_end = round(total * 0.85)
    return max(1, min(train_end, total - 2)), max(2, min(validation_end, total - 1))


def _adjust_equal_timestamp_boundary(ordered: pd.DataFrame, boundary: int) -> tuple[int, bool]:
    if boundary <= 0 or boundary >= len(ordered) or ordered.loc[boundary - 1, "_parsed_author_date"] != ordered.loc[boundary, "_parsed_author_date"]:
        return boundary, False
    timestamp = ordered.loc[boundary, "_parsed_author_date"]
    start = boundary
    end = boundary
    while start > 0 and ordered.loc[start - 1, "_parsed_author_date"] == timestamp:
        start -= 1
    while end < len(ordered) and ordered.loc[end, "_parsed_author_date"] == timestamp:
        end += 1
    candidates = [candidate for candidate in (start, end) if 1 <= candidate < len(ordered)]
    chosen = min(candidates, key=lambda candidate: (abs(candidate - boundary), candidate))
    return chosen, True


def project_wise_split(metadata: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    ordered = metadata.sort_values(["project", "_parsed_author_date", "_source_row_id"], kind="mergesort").reset_index(drop=True)
    splits = {name: [] for name in ("train", "validation", "test")}
    project_info: dict[str, dict[str, Any]] = {}
    output_rank = 0
    for project, project_frame in ordered.groupby("project", sort=True):
        project_frame = project_frame.reset_index(drop=True)
        total = len(project_frame)
        train_end, validation_end = _allocation(total)
        train_end, train_tie = _adjust_equal_timestamp_boundary(project_frame, train_end)
        validation_end, validation_tie = _adjust_equal_timestamp_boundary(project_frame, validation_end)
        if total >= 3:
            train_end = min(train_end, total - 2)
            validation_end = min(validation_end, total - 1)
            if validation_end <= train_end:
                validation_end = train_end + 1
        project_frame["_project_rank"] = range(total)
        project_frame["_output_rank"] = range(output_rank, output_rank + total)
        output_rank += total
        pieces = {"train": project_frame.iloc[:train_end].copy(), "validation": project_frame.iloc[train_end:validation_end].copy(), "test": project_frame.iloc[validation_end:].copy()}
        for name, piece in pieces.items():
            splits[name].append(piece)
        project_info[str(project)] = {"total": total, "train": len(pieces["train"]), "validation": len(pieces["validation"]), "test": len(pieces["test"]), "date_range": {"min": project_frame["_parsed_author_date"].min().isoformat(), "max": project_frame["_parsed_author_date"].max().isoformat()}, "boundary_equal_timestamp": bool(train_tie or validation_tie), "small_project": total < 3}
    combined = {name: pd.concat(parts, ignore_index=True) if parts else ordered.iloc[0:0].copy() for name, parts in splits.items()}

    # Every source row must belong to exactly one partition.
    assigned = pd.concat(
        [part["_source_row_id"] for part in combined.values()],
        ignore_index=True,
    )
    if len(assigned) != len(metadata) or assigned.nunique() != len(metadata):
        raise ValueError("Project-wise allocation did not assign every source row exactly once")

    return combined, project_info


def _distribution(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {str(key): int(value) for key, value in frame[column].value_counts().sort_index().items()}


def _cross_distribution(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame.groupby(["language", "buggy"], sort=True).size()
    return {f"{language}|buggy={buggy}": int(count) for (language, buggy), count in counts.items()}


def distributions(frame: pd.DataFrame) -> dict[str, Any]:
    return {"rows": len(frame), "language": _distribution(frame, "language"), "buggy": _distribution(frame, "buggy"), "language_buggy": _cross_distribution(frame), "project": _distribution(frame, "project")}


def _proportions(counts: dict[str, int], rows: int) -> dict[str, float]:
    return {key: value / rows if rows else 0.0 for key, value in counts.items()}


def _date_range(frame: pd.DataFrame) -> dict[str, str | None]:
    if frame.empty:
        return {"min": None, "max": None}
    return {"min": frame["_parsed_author_date"].min().isoformat(), "max": frame["_parsed_author_date"].max().isoformat()}


def analyze_and_warn(splits: dict[str, pd.DataFrame], full: pd.DataFrame, project_info: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    full_dist = distributions(full)
    report_rows: list[dict[str, Any]] = []
    warnings_found: list[dict[str, Any]] = []
    for name, part in [("overall", full), *splits.items()]:
        dist = distributions(part)
        date_range = _date_range(part)
        print(f"\n{name.upper()}\nrows: {len(part)}\ndate range: {date_range['min']} -> {date_range['max']}")
        for language in LANGUAGES:
            print(f"{language}: {_proportions(dist['language'], len(part)).get(language, 0.0):.2%}")
        buggy_props = _proportions(dist["buggy"], len(part))
        for label in (0, 1):
            print(f"buggy={label}: {buggy_props.get(str(label), 0.0):.2%}")
        for category in ("language", "buggy", "language_buggy", "project"):
            expected = _proportions(full_dist[category], len(full))
            actual = _proportions(dist[category], len(part))
            for item in sorted(set(expected) | set(actual)):
                difference = actual.get(item, 0.0) - expected.get(item, 0.0)
                report_rows.append({"split": name, "category": category, "item": item, "count": dist[category].get(item, 0), "proportion": actual.get(item, 0.0), "full_proportion": expected.get(item, 0.0), "difference": difference})
                threshold = 0.05 if category in ("language", "buggy", "language_buggy") else 0.10
                if name != "overall" and abs(difference) > threshold:
                    warnings_found.append({"split": name, "category": category, "item": item, "difference": difference})
    for project, info in project_info.items():
        report_rows.append({"split": "project_summary", "category": "project_split", "item": project, "count": info["total"], "proportion": None, "full_proportion": None, "difference": None, "train": info["train"], "validation": info["validation"], "test": info["test"], "date_min": info["date_range"]["min"], "date_max": info["date_range"]["max"], "small_project": info["small_project"], "boundary_equal_timestamp": info["boundary_equal_timestamp"]})
        if info["validation"] == 0 or info["test"] == 0:
            warnings_found.append({"project": project, "issue": "zero validation/test observations", "details": info})
    for item in warnings_found:
        warnings.warn(f"Split distribution warning: {item}", UserWarning)
    return {name: distributions(part) for name, part in splits.items()}, report_rows


def assert_metadata_splits(splits: dict[str, pd.DataFrame], total_rows: int, project_info: dict[str, dict[str, Any]]) -> None:
    assert sum(len(part) for part in splits.values()) == total_rows
    ids = [set(part["commit_id"]) for part in splits.values()]
    union = set.union(*ids)
    assert len(union) == total_rows
    assert sum(len(item) for item in ids) == len(union), "Commit IDs overlap across splits"
    for project, info in project_info.items():
        project_parts = {name: part[part["project"] == project] for name, part in splits.items()}
        assert {name: len(part) for name, part in project_parts.items()} == {name: info[name] for name in ("train", "validation", "test")}
        if info["train"] and info["validation"]:
            assert project_parts["train"]["_parsed_author_date"].max() <= project_parts["validation"]["_parsed_author_date"].min()
        if info["validation"] and info["test"]:
            assert project_parts["validation"]["_parsed_author_date"].max() <= project_parts["test"]["_parsed_author_date"].min()


def write_chunked_outputs(input_path: Path, output_dir: Path, original_columns: list[str], splits: dict[str, pd.DataFrame], chunk_size: int) -> dict[str, int]:
    """Copy full rows by source-row ID into bounded chronological buckets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source_to_destination: dict[int, tuple[str, int]] = {}
    for split_name, part in splits.items():
        for source_id, rank in zip(part["_source_row_id"], part["_output_rank"]):
            source_to_destination[int(source_id)] = (split_name, int(rank))
    output_counts = {name: 0 for name in splits}
    with tempfile.TemporaryDirectory(prefix="chronological_split_", dir=output_dir) as temp_name:
        temp_dir = Path(temp_name)
        bucket_paths: dict[tuple[str, int], Path] = {}
        first_write: set[tuple[str, int]] = set()
        source_row_id = 0
        for chunk in pd.read_csv(input_path, dtype=str, keep_default_na=False, chunksize=chunk_size):
            if list(chunk.columns) != original_columns:
                raise ValueError("CSV columns changed between first and second pass")
            destinations = [source_to_destination.get(source_row_id + offset) for offset in range(len(chunk))]
            chunk["__destination"] = destinations
            selected = chunk[chunk["__destination"].notna()].copy()
            if selected.empty:
                source_row_id += len(chunk)
                continue
            selected["__split"] = selected["__destination"].map(lambda item: item[0])
            selected["__output_rank"] = selected["__destination"].map(lambda item: item[1])
            for split_name in splits:
                split_rows = selected[selected["__split"] == split_name]
                for bucket, bucket_rows in split_rows.groupby(split_rows["__output_rank"] // chunk_size, sort=True):
                    key = (split_name, int(bucket))
                    path = bucket_paths.setdefault(key, temp_dir / f"{split_name}_{bucket}.csv")
                    bucket_rows.to_csv(path, mode="a", header=key not in first_write, index=False)
                    first_write.add(key)
            source_row_id += len(chunk)
        for split_name in splits:
            output_path = output_dir / f"{split_name}.csv"
            if output_path.exists():
                output_path.unlink()
            for _, bucket_path in sorted((bucket, path) for (split, bucket), path in bucket_paths.items() if split == split_name):
                for bucket in pd.read_csv(bucket_path, dtype=str, keep_default_na=False, chunksize=chunk_size):
                    bucket = bucket.sort_values("__output_rank", kind="mergesort")
                    bucket[original_columns].to_csv(output_path, mode="a", header=not output_path.exists(), index=False)
                    output_counts[split_name] += len(bucket)
    return output_counts


def build_metadata(splits: dict[str, pd.DataFrame], project_info: dict[str, dict[str, Any]], total_rows: int, methods: dict[str, int], duplicate_ids: int, duplicate_rows: int, duplicate_policy: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {"total_rows": total_rows, "duplicate_commit_ids_found": duplicate_ids, "duplicate_commit_rows_found": duplicate_rows, "duplicate_policy": duplicate_policy, "date_parsing": {"representations": methods, "sanity_lower_bound": SANITY_LOWER_BOUND.isoformat(), "sanity_upper_bound": SANITY_UPPER_BOUND.isoformat()}, "project_split_policy": "Each project sorted by parsed author_date then _source_row_id; oldest 70% train, next 15% validation, latest 15% test.", "small_project_policy": "Projects with at least three rows receive nonempty three-way partitions. Projects with fewer than three rows are assigned deterministically to the earliest possible partitions without duplication and are explicitly flagged.", "equal_timestamp_policy": "Use _source_row_id for deterministic ties; if a boundary falls inside an equal-timestamp group, move it to the nearest group edge when possible and record the adjustment.", "projects": project_info, "splits": {}}
    for name, part in splits.items():
        metadata["splits"][name] = {"rows": len(part), "percentage": len(part) / total_rows, "date_range": _date_range(part), **distributions(part)}
    metadata["embeddings_copied_without_transformation"] = True
    metadata["pca_performed"] = False
    return metadata


def _hash_commit_embedding_pairs(path: Path, chunk_size: int) -> str:
    """Compute an order-independent digest of commit_id + raw embedding strings.

    Embeddings are streamed in chunks. Only compact 32-byte row digests are
    retained, so the full embedding column is never loaded into memory.
    """

    row_digests: list[bytes] = []
    for chunk in pd.read_csv(
        path,
        usecols=["commit_id", "embedding"],
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
    ):
        for commit_id, embedding in zip(chunk["commit_id"], chunk["embedding"]):
            row_digests.append(
                hashlib.sha256(
                    (str(commit_id) + "\0" + str(embedding)).encode("utf-8")
                ).digest()
            )

    aggregate = hashlib.sha256()
    for row_digest in sorted(row_digests):
        aggregate.update(row_digest)
    return aggregate.hexdigest()


def validate_output_integrity(
    input_path: Path,
    output_dir: Path,
    original_columns: list[str],
    expected_counts: dict[str, int],
    chunk_size: int,
) -> None:
    """Validate headers, row counts, commit coverage, and raw embedding preservation."""
    output_files = {
        name: output_dir / f"{name}.csv"
        for name in ("train", "validation", "test")
    }

    for name, path in output_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing output file: {path}")
        header = list(pd.read_csv(path, nrows=0).columns)
        if header != original_columns:
            raise ValueError(
                f"{name}.csv columns differ from the original input columns"
            )

    # Streaming commit-ID accounting avoids loading the embedding column.
    input_ids: set[str] = set()
    output_ids: set[str] = set()

    for chunk in pd.read_csv(
        input_path,
        usecols=["commit_id"],
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
    ):
        input_ids.update(chunk["commit_id"].tolist())

    for name, path in output_files.items():
        count = 0
        for chunk in pd.read_csv(
            path,
            usecols=["commit_id"],
            dtype=str,
            keep_default_na=False,
            chunksize=chunk_size,
        ):
            count += len(chunk)
            output_ids.update(chunk["commit_id"].tolist())
        if count != expected_counts[name]:
            raise ValueError(
                f"{name}.csv row count {count} != expected {expected_counts[name]}"
            )

    if len(input_ids) != sum(expected_counts.values()):
        raise ValueError(
            f"Input contains {len(input_ids)} unique commit IDs but "
            f"{sum(expected_counts.values())} logical rows were expected"
        )
    if len(output_ids) != len(input_ids):
        raise ValueError("Output commit IDs contain duplicates or missing coverage")
    if output_ids != input_ids:
        missing = len(input_ids - output_ids)
        extra = len(output_ids - input_ids)
        raise ValueError(
            f"Output commit-ID coverage mismatch: missing={missing}, extra={extra}"
        )

    if "embedding" not in original_columns:
        raise ValueError(
            "Expected an 'embedding' column for preservation validation, "
            "but it is missing from the input."
        )

    input_digest = _hash_commit_embedding_pairs(input_path, chunk_size)

    # Compute the same order-independent digest across all three output files.
    # Each helper call returns a digest of one file, so we need a combined
    # row-level digest rather than hashing the three file digests. Re-read the
    # three files in chunks and retain only compact row hashes.
    output_row_digests: list[bytes] = []
    for name in ("train", "validation", "test"):
        for chunk in pd.read_csv(
            output_files[name],
            usecols=["commit_id", "embedding"],
            dtype=str,
            keep_default_na=False,
            chunksize=chunk_size,
        ):
            for commit_id, embedding in zip(chunk["commit_id"], chunk["embedding"]):
                output_row_digests.append(
                    hashlib.sha256(
                        (str(commit_id) + "\0" + str(embedding)).encode("utf-8")
                    ).digest()
                )

    combined_output_digest = hashlib.sha256()
    for row_digest in sorted(output_row_digests):
        combined_output_digest.update(row_digest)

    if combined_output_digest.hexdigest() != input_digest:
        raise ValueError(
            "Embedding preservation check failed: output commit_id+embedding "
            "content differs from the input."
        )

    print("Output header check: passed")
    print("Output row/commit coverage check: passed")
    print("Embedding preservation check: passed")




def run(input_path: Path = INPUT_PATH, output_dir: Path = OUTPUT_DIR, duplicate_policy: str = "error", chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
    metadata_all, original_columns, date_methods = load_metadata(input_path, chunk_size)
    metadata, duplicate_ids, duplicate_rows = audit_duplicates(metadata_all, duplicate_policy)
    splits, project_info = project_wise_split(metadata)
    distributions_by_split, report_rows = analyze_and_warn(splits, metadata, project_info)
    assert_metadata_splits(splits, len(metadata), project_info)
    output_counts = write_chunked_outputs(input_path, output_dir, original_columns, splits, chunk_size)
    expected_counts = {name: len(part) for name, part in splits.items()}
    assert output_counts == expected_counts, f"Second-pass output counts differ: {output_counts} != {expected_counts}"
    validate_output_integrity(input_path, output_dir, original_columns, expected_counts, chunk_size)
    metadata_json = build_metadata(splits, project_info, len(metadata), date_methods, duplicate_ids, duplicate_rows, duplicate_policy)
    metadata_json["distributions"] = distributions_by_split
    metadata_json["second_pass_row_counts"] = output_counts
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split_metadata.json").write_text(json.dumps(metadata_json, indent=2), encoding="utf-8")
    pd.DataFrame(report_rows).to_csv(output_dir / "split_distribution_report.csv", index=False)
    print(f"\nTotal logical rows: {len(metadata)}")
    for name in ("train", "validation", "test"):
        print(f"{name.title()} rows: {len(splits[name])}")
        print(f"{name.title()} date range: {_date_range(splits[name])['min']} -> {_date_range(splits[name])['max']}")
        print(f"{name.title()} language distribution: {distributions_by_split[name]['language']}")
        print(f"{name.title()} buggy distribution: {distributions_by_split[name]['buggy']}")
        print(f"{name.title()} output: {output_dir / f'{name}.csv'}")
    print(f"Duplicate commit IDs: {duplicate_ids}")
    print(f"Validation report: {output_dir / 'split_distribution_report.csv'}")
    print("Project-wise chronological split completed successfully.")
    print("No commit overlap detected.")
    print("Embeddings copied without transformation.")
    print("No PCA, scaling, normalization, or random sampling performed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--duplicate-policy", choices=("drop", "error"), default="error")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    run(args.input, args.output_dir, args.duplicate_policy, args.chunk_size)


if __name__ == "__main__":
    main()
