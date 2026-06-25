from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from git import Repo
from pydriller import Repository

COLUMNS = [
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

MAX_COMMITS_PER_REPO = 2000

BUG_KEYWORDS = ("fix", "bug", "error", "issue", "patch", "resolve", "hotfix")
PY_EXTS = {".py"}
CPP_EXTS = {".cpp", ".cc", ".hpp", ".h"}


@dataclass(frozen=True)
class RepoSpec:
    name: str
    path: Path
    project: str
    extensions: Set[str]


class FileHistoryTracker:
    def __init__(self, repo_path: Path) -> None:
        self.repo = Repo(str(repo_path))
        self.file_last_commit: Dict[str, datetime] = {}
        self.file_developers: Dict[str, Set[str]] = defaultdict(set)
        self.file_commit_counts: Dict[str, int] = defaultdict(int)
        self.author_commit_count: Dict[str, int] = defaultdict(int)
        self.author_recent_commits: Dict[str, deque[datetime]] = defaultdict(deque)
        self.author_subsystem_count: Dict[Tuple[str, str], int] = defaultdict(int)

    def _norm(self, path: str) -> str:
        return path.replace("\\", "/").lower()

    def compute_metrics(self, author: str, commit_dt: datetime, relevant_files: Sequence[str]) -> Dict[str, float]:
        file_ages = []
        unique_developers: Set[str] = set()
        prior_file_commits = 0
        touched_subsystems: Set[str] = set()

        for path in relevant_files:
            normalized = self._norm(path)
            last_commit = self.file_last_commit.get(normalized)
            if last_commit is not None:
                file_ages.append(max((commit_dt - last_commit).total_seconds(), 0.0))
                unique_developers.update(self.file_developers.get(normalized, set()))
                prior_file_commits += self.file_commit_counts.get(normalized, 0)
            else:
                file_ages.append(0.0)

            subsystem = Path(normalized).parts[0] if Path(normalized).parts else ""
            touched_subsystems.add(subsystem)

        recent_commits = self.author_recent_commits.get(author, deque())
        cutoff = commit_dt.timestamp() - 365 * 24 * 3600
        recent_count = sum(1 for dt in recent_commits if dt.timestamp() >= cutoff)
        subsystem_experience = sum(self.author_subsystem_count.get((author, subsystem), 0) for subsystem in touched_subsystems)

        return {
            "ndev": float(len(unique_developers)),
            "age": float(np.mean(file_ages)) if file_ages else 0.0,
            "nuc": float(prior_file_commits),
            "aexp": float(self.author_commit_count.get(author, 0)),
            "arexp": float(recent_count),
            "asexp": float(subsystem_experience),
        }

    def record_commit(self, author: str, commit_dt: datetime, changed_files: Sequence[str]) -> None:
        self.author_commit_count[author] += 1
        recent = self.author_recent_commits[author]
        recent.append(commit_dt)
        cutoff = commit_dt.timestamp() - 365 * 24 * 3600
        while recent and recent[0].timestamp() < cutoff:
            recent.popleft()

        for path in changed_files:
            normalized = self._norm(path)
            self.file_commit_counts[normalized] += 1
            self.file_last_commit[normalized] = commit_dt
            self.file_developers[normalized].add(author)
        touched_subsystems = {Path(self._norm(path)).parts[0] if Path(self._norm(path)).parts else "" for path in changed_files}
        for subsystem in touched_subsystems:
            self.author_subsystem_count[(author, subsystem)] += 1


def commit_is_buggy(message: str) -> bool:
    text = message.lower()
    return any(keyword in text for keyword in BUG_KEYWORDS)


def normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def git_commit_list(repo_path: Path, limit: int) -> List[str]:
    repo = Repo(str(repo_path))
    return [commit.hexsha for commit in repo.iter_commits(max_count=limit)]


def file_is_relevant(path: str, extensions: Set[str]) -> bool:
    suffix = Path(path).suffix.lower()
    return suffix in extensions


def touched_relevant_files(commit, extensions: Set[str]) -> List[str]:
    files: List[str] = []
    for modified in commit.modified_files:
        if modified.new_path and file_is_relevant(modified.new_path, extensions):
            files.append(normalize_path(modified.new_path))
        elif modified.old_path and file_is_relevant(modified.old_path, extensions):
            files.append(normalize_path(modified.old_path))
    return sorted(set(files))


def directories_touched(paths: Iterable[str]) -> int:
    dirs = set()
    for path in paths:
        parent = Path(path).parent.as_posix()
        if parent and parent != ".":
            dirs.add(parent)
    return len(dirs)


def subsystems_touched(paths: Iterable[str]) -> int:
    subsystems = set()
    for path in paths:
        parts = Path(path).parts
        if parts:
            subsystems.add(parts[0])
    return len(subsystems)


def entropy_from_paths(commit, relevant_files: Sequence[str]) -> float:
    if not relevant_files:
        return 0.0

    changes = []
    total = 0
    for modified in commit.modified_files:
        path = modified.new_path or modified.old_path
        if not path:
            continue
        path = normalize_path(path)
        if path not in relevant_files:
            continue
        additions = int(getattr(modified, "added_lines", 0) or 0)
        deletions = int(getattr(modified, "deleted_lines", 0) or 0)
        count = additions + deletions
        if count <= 0:
            count = 1
        changes.append(count)
        total += count

    if total <= 0:
        return 0.0

    entropy = 0.0
    for count in changes:
        p_i = count / total
        if p_i > 0:
            entropy -= p_i * math.log2(p_i)
    return float(entropy)


def average_seconds(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def commit_to_row(commit, spec: RepoSpec, history: FileHistoryTracker) -> Dict[str, object] | None:
    relevant_files = touched_relevant_files(commit, spec.extensions)
    if not relevant_files:
        return None

    message = commit.msg or ""
    buggy = commit_is_buggy(message)
    commit_dt = commit.author_date.replace(tzinfo=timezone.utc) if commit.author_date.tzinfo is None else commit.author_date.astimezone(timezone.utc)
    author = commit.author.name or commit.author.email or "unknown"
    metrics = history.compute_metrics(author, commit_dt, relevant_files)
    diff_parts = []
    for modified in commit.modified_files:
        path = modified.new_path or modified.old_path
        if not path:
            continue
        if normalize_path(path) not in relevant_files:
            continue
        patch = getattr(modified, "diff", None) or getattr(modified, "source_code", None) or ""
        diff_parts.append(f"--- {normalize_path(path)}\n{patch}")

    total_added = int(sum(int(getattr(m, "added_lines", 0) or 0) for m in commit.modified_files if normalize_path((m.new_path or m.old_path or "")) in relevant_files))
    total_deleted = int(sum(int(getattr(m, "deleted_lines", 0) or 0) for m in commit.modified_files if normalize_path((m.new_path or m.old_path or "")) in relevant_files))

    row = {
        "commit_id": commit.hash,
        "project": spec.project,
        "buggy": bool(buggy),
        "fix": bool(buggy),
        "year": int(commit_dt.year),
        "author_date": int(commit_dt.timestamp()),
        "la": total_added,
        "ld": total_deleted,
        "nf": int(len(relevant_files)),
        "nd": int(directories_touched(relevant_files)),
        "ns": int(subsystems_touched(relevant_files)),
        "ent": float(entropy_from_paths(commit, relevant_files)),
        "ndev": int(metrics["ndev"]),
        "age": float(metrics["age"]),
        "nuc": int(metrics["nuc"]),
        "aexp": int(metrics["aexp"]),
        "arexp": int(metrics["arexp"]),
        "asexp": int(metrics["asexp"]),
        "message": message,
        "diff": "\n\n".join(diff_parts),
    }
    return row


def mine_repository(spec: RepoSpec, max_commits: int = MAX_COMMITS_PER_REPO) -> pd.DataFrame:
    history = FileHistoryTracker(spec.path)
    commit_hashes = list(reversed(git_commit_list(spec.path, max_commits)))
    rows: List[Dict[str, object]] = []

    for index, commit in enumerate(Repository(str(spec.path), only_commits=commit_hashes).traverse_commits(), start=1):
        row = commit_to_row(commit, spec, history)
        changed_files = [normalize_path(m.new_path or m.old_path) for m in commit.modified_files if (m.new_path or m.old_path)]
        history.record_commit(commit.author.name or commit.author.email or "unknown", commit.author_date.replace(tzinfo=timezone.utc) if commit.author_date.tzinfo is None else commit.author_date.astimezone(timezone.utc), changed_files)
        if row is not None:
            rows.append(row)
        if index % 100 == 0:
            print(f"{spec.name}: processed {index}/{len(commit_hashes)} commits, kept {len(rows)} rows", flush=True)

    df = pd.DataFrame(rows, columns=COLUMNS)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    print(df.columns.tolist())
    print(df.shape)
    print(df.head())
    if list(df.columns) != COLUMNS:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    df.to_csv(output_path, index=False)


def load_optional_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    flask_repo = base_dir / "flask"
    opencv_repo = base_dir / "opencv"

    specs = [
        RepoSpec(name="flask", path=flask_repo, project="python/flask", extensions=PY_EXTS),
        RepoSpec(name="opencv", path=opencv_repo, project="cpp/opencv", extensions=CPP_EXTS),
    ]

    outputs = []
    for spec in specs:
        print(f"Mining {spec.name}...")
        df = mine_repository(spec, max_commits=MAX_COMMITS_PER_REPO)
        validate_schema(df)
        output_path = base_dir / f"{spec.project.replace('/', '_')}_dataset.csv"
        save_dataset(df, output_path)
        outputs.append(output_path)
        print(f"Saved {output_path} with {len(df)} rows")

    java_path = base_dir / "java_dataset.csv"
    java_df = load_optional_csv(java_path)
    if java_df is not None:
        python_df = pd.read_csv(outputs[0])
        cpp_df = pd.read_csv(outputs[1])
        for frame_name, frame in (("java", java_df), ("python", python_df), ("cpp", cpp_df)):
            if list(frame.columns) != COLUMNS:
                raise ValueError(f"{frame_name} dataset schema mismatch: {list(frame.columns)}")
        final_df = pd.concat([java_df, python_df, cpp_df], ignore_index=True)
        output_path = Path(__file__).resolve().parent.parent / "data" / "final_multilanguage_dataset.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_path, index=False)
        print(final_df.shape)
        print(final_df["buggy"].value_counts(dropna=False))
        print(final_df.isna().sum())
        print(f"Saved {output_path}")
    else:
        print("java_dataset.csv not found; skipped final merge.")


if __name__ == "__main__":
    main()
