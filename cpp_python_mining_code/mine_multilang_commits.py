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
    "language",
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

TARGET_COMMITS_PER_LANGUAGE = 12000

BUG_KEYWORDS = ("fix", "bug", "error", "issue", "patch", "resolve", "hotfix")
PY_EXTS = {".py"}
CPP_EXTS = {".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx", ".h"}


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
        "language": (
            "python"
            if spec.project.startswith("python/")
            else "cpp"
        ),
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


def recent_commit_window_for_target(
    repo_path: Path,
    extensions: Set[str],
    target_relevant: int,
) -> List[str]:
    """
    Scan Git history from newest to oldest until target_relevant commits that
    touch the requested language are found. Return the whole scanned window
    in chronological order (oldest -> newest) so JIT history features are
    computed without future leakage inside the selected window.
    """
    repo = Repo(str(repo_path))
    newest_to_oldest: List[str] = []
    relevant_found = 0

    for inspected, git_commit in enumerate(repo.iter_commits(), start=1):
        newest_to_oldest.append(git_commit.hexsha)

        try:
            changed_paths = list(git_commit.stats.files.keys())
        except Exception:
            changed_paths = []

        if any(file_is_relevant(path, extensions) for path in changed_paths):
            relevant_found += 1

        if inspected % 500 == 0:
            print(
                f"Pre-scan: inspected {inspected} Git commits; "
                f"found {relevant_found}/{target_relevant} relevant commits",
                flush=True,
            )

        if relevant_found >= target_relevant:
            break

    if relevant_found < target_relevant:
        print(
            f"WARNING: repository history contains only {relevant_found} "
            f"commits matching {sorted(extensions)}; requested {target_relevant}.",
            flush=True,
        )

    return list(reversed(newest_to_oldest))


def mine_repository(
    spec: RepoSpec,
    target_rows: int,
) -> pd.DataFrame:
    """
    Mine the most recent target_rows relevant commits for one repository.

    First determine a recent commit window, then traverse that window
    chronologically so FileHistoryTracker only uses prior commits.
    """
    if target_rows <= 0:
        return pd.DataFrame(columns=COLUMNS)

    commit_hashes = recent_commit_window_for_target(
        spec.path,
        spec.extensions,
        target_rows,
    )

    history = FileHistoryTracker(spec.path)
    rows: List[Dict[str, object]] = []

    for index, commit in enumerate(
        Repository(
            str(spec.path),
            only_commits=commit_hashes,
        ).traverse_commits(),
        start=1,
    ):
        row = commit_to_row(commit, spec, history)

        changed_files = [
            normalize_path(m.new_path or m.old_path)
            for m in commit.modified_files
            if (m.new_path or m.old_path)
        ]

        commit_dt = (
            commit.author_date.replace(tzinfo=timezone.utc)
            if commit.author_date.tzinfo is None
            else commit.author_date.astimezone(timezone.utc)
        )

        history.record_commit(
            commit.author.name or commit.author.email or "unknown",
            commit_dt,
            changed_files,
        )

        if row is not None:
            rows.append(row)

        if index % 100 == 0:
            print(
                f"{spec.name}: processed {index}/{len(commit_hashes)} Git commits; "
                f"retained {len(rows)}/{target_rows}",
                flush=True,
            )

    # Keep the newest target_rows relevant rows if the pre-scan slightly
    # over-counted due to Git/PyDriller path representation differences.
    if len(rows) > target_rows:
        rows = rows[-target_rows:]

    return pd.DataFrame(rows, columns=COLUMNS)


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


def find_existing_flask_csv(base_dir: Path) -> Path:
    candidates = [
        base_dir / "python_flask_dataset.csv",
        base_dir / "flask_dataset.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Existing Flask CSV not found. Expected python_flask_dataset.csv "
        "or flask_dataset.csv beside this script."
    )


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    flask_path = find_existing_flask_csv(base_dir)
    flask_df = pd.read_csv(flask_path)

    if list(flask_df.columns) != COLUMNS:
        raise ValueError(
            f"Flask dataset schema mismatch: {list(flask_df.columns)}"
        )

    # Avoid accidental duplicate commit IDs in an already-mined Flask file.
    flask_df = flask_df.drop_duplicates(subset=["commit_id"]).copy()

    flask_rows = len(flask_df)
    remaining = max(TARGET_COMMITS_PER_LANGUAGE - flask_rows, 0)

    print("=" * 72)
    print("PYTHON MINING TARGET")
    print("=" * 72)
    print(f"Existing Flask rows: {flask_rows}")
    print(f"Python target:       {TARGET_COMMITS_PER_LANGUAGE}")
    print(f"Still required:      {remaining}")

    if remaining == 0:
        final_python = flask_df.iloc[:TARGET_COMMITS_PER_LANGUAGE].copy()
    else:
        # Split the missing amount approximately evenly. If Django yields
        # fewer than requested, CPython automatically receives the shortfall.
        django_target = (remaining + 1) // 2

        django_spec = RepoSpec(
            name="django",
            path=base_dir / "django",
            project="python/django",
            extensions=PY_EXTS,
        )
        cpython_spec = RepoSpec(
            name="cpython",
            path=base_dir / "cpython",
            project="python/cpython",
            extensions=PY_EXTS,
        )

        for spec in (django_spec, cpython_spec):
            if not spec.path.exists():
                raise FileNotFoundError(
                    f"Repository not found: {spec.path}\n"
                    f"Clone it beside this script before running."
                )

        print(f"\nMining Django target: {django_target}")
        django_df = mine_repository(
            django_spec,
            target_rows=django_target,
        )
        validate_schema(django_df)
        django_path = base_dir / "python_django_dataset.csv"
        save_dataset(django_df, django_path)
        print(f"Saved {django_path} with {len(django_df)} rows")

        cpython_target = remaining - len(django_df)
        print(f"\nMining CPython target: {cpython_target}")
        cpython_df = mine_repository(
            cpython_spec,
            target_rows=cpython_target,
        )
        validate_schema(cpython_df)
        cpython_path = base_dir / "python_cpython_dataset.csv"
        save_dataset(cpython_df, cpython_path)
        print(f"Saved {cpython_path} with {len(cpython_df)} rows")

        final_python = pd.concat(
            [flask_df, django_df, cpython_df],
            ignore_index=True,
        )

        # Cross-repository hashes should already be distinct in practice,
        # but project + commit_id is the safe uniqueness key.
        final_python = final_python.drop_duplicates(
            subset=["project", "commit_id"]
        )

        if len(final_python) > TARGET_COMMITS_PER_LANGUAGE:
            # Preserve all Flask rows already produced and trim only if a
            # pre-scan yielded a tiny excess.
            final_python = final_python.iloc[:TARGET_COMMITS_PER_LANGUAGE]

    output_path = base_dir / "python_12000_dataset.csv"
    save_dataset(final_python, output_path)

    print("\n" + "=" * 72)
    print("FINAL PYTHON DATASET")
    print("=" * 72)
    print(f"Rows: {len(final_python)}")
    print(final_python["project"].value_counts())
    print("\nBug/fix labels:")
    print(final_python["buggy"].value_counts(dropna=False))
    print("\nMissing values:")
    print(final_python.isna().sum())
    print(f"\nSaved: {output_path}")

    if len(final_python) < TARGET_COMMITS_PER_LANGUAGE:
        raise RuntimeError(
            f"Only {len(final_python)} Python rows were produced; "
            f"{TARGET_COMMITS_PER_LANGUAGE} were required."
        )


if __name__ == "__main__":
    main()