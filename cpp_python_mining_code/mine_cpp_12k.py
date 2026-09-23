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

TARGET_ROWS = 12000

BUG_KEYWORDS = (
    "fix",
    "bug",
    "error",
    "issue",
    "patch",
    "resolve",
    "hotfix",
)

CPP_EXTS = {
    ".cpp",
    ".cc",
    ".cxx",
    ".hpp",
    ".hh",
    ".hxx",
    ".h",
}

CHECKPOINT_EVERY = 100


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

    def compute_metrics(
        self,
        author: str,
        commit_dt: datetime,
        relevant_files: Sequence[str],
    ) -> Dict[str, float]:

        file_ages = []
        unique_developers: Set[str] = set()
        prior_file_commits = 0
        touched_subsystems: Set[str] = set()

        for path in relevant_files:
            normalized = self._norm(path)
            last_commit = self.file_last_commit.get(normalized)

            if last_commit is not None:
                file_ages.append(
                    max((commit_dt - last_commit).total_seconds(), 0.0)
                )
                unique_developers.update(
                    self.file_developers.get(normalized, set())
                )
                prior_file_commits += self.file_commit_counts.get(
                    normalized, 0
                )
            else:
                file_ages.append(0.0)

            subsystem = (
                Path(normalized).parts[0]
                if Path(normalized).parts
                else ""
            )
            touched_subsystems.add(subsystem)

        recent_commits = self.author_recent_commits.get(
            author,
            deque(),
        )

        cutoff = commit_dt.timestamp() - 365 * 24 * 3600

        recent_count = sum(
            1
            for dt in recent_commits
            if dt.timestamp() >= cutoff
        )

        subsystem_experience = sum(
            self.author_subsystem_count.get(
                (author, subsystem),
                0,
            )
            for subsystem in touched_subsystems
        )

        return {
            "ndev": float(len(unique_developers)),
            "age": float(np.mean(file_ages)) if file_ages else 0.0,
            "nuc": float(prior_file_commits),
            "aexp": float(self.author_commit_count.get(author, 0)),
            "arexp": float(recent_count),
            "asexp": float(subsystem_experience),
        }

    def record_commit(
        self,
        author: str,
        commit_dt: datetime,
        changed_files: Sequence[str],
    ) -> None:

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

        touched_subsystems = {
            Path(self._norm(path)).parts[0]
            if Path(self._norm(path)).parts
            else ""
            for path in changed_files
        }

        for subsystem in touched_subsystems:
            self.author_subsystem_count[
                (author, subsystem)
            ] += 1


def commit_is_buggy(message: str) -> bool:
    text = message.lower()
    return any(keyword in text for keyword in BUG_KEYWORDS)


def normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def file_is_relevant(
    path: str,
    extensions: Set[str],
) -> bool:
    suffix = Path(path).suffix.lower()
    return suffix in extensions


def touched_relevant_files(
    commit,
    extensions: Set[str],
) -> List[str]:

    files = []

    for modified in commit.modified_files:

        if (
            modified.new_path
            and file_is_relevant(
                modified.new_path,
                extensions,
            )
        ):
            files.append(
                normalize_path(modified.new_path)
            )

        elif (
            modified.old_path
            and file_is_relevant(
                modified.old_path,
                extensions,
            )
        ):
            files.append(
                normalize_path(modified.old_path)
            )

    return sorted(set(files))


def directories_touched(
    paths: Iterable[str],
) -> int:

    dirs = set()

    for path in paths:
        parent = Path(path).parent.as_posix()

        if parent and parent != ".":
            dirs.add(parent)

    return len(dirs)


def subsystems_touched(
    paths: Iterable[str],
) -> int:

    subsystems = set()

    for path in paths:
        parts = Path(path).parts

        if parts:
            subsystems.add(parts[0])

    return len(subsystems)


def entropy_from_paths(
    commit,
    relevant_files: Sequence[str],
) -> float:

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

        additions = int(
            getattr(modified, "added_lines", 0) or 0
        )

        deletions = int(
            getattr(modified, "deleted_lines", 0) or 0
        )

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


def directories_touched_unused(paths):
    return directories_touched(paths)


def commit_to_row(
    commit,
    spec: RepoSpec,
    history: FileHistoryTracker,
):

    relevant_files = touched_relevant_files(
        commit,
        spec.extensions,
    )

    if not relevant_files:
        return None

    message = commit.msg or ""

    buggy = commit_is_buggy(message)

    commit_dt = (
        commit.author_date.replace(tzinfo=timezone.utc)
        if commit.author_date.tzinfo is None
        else commit.author_date.astimezone(timezone.utc)
    )

    author = (
        commit.author.name
        or commit.author.email
        or "unknown"
    )

    metrics = history.compute_metrics(
        author,
        commit_dt,
        relevant_files,
    )

    diff_parts = []

    for modified in commit.modified_files:

        path = modified.new_path or modified.old_path

        if not path:
            continue

        if normalize_path(path) not in relevant_files:
            continue

        patch = (
            getattr(modified, "diff", None)
            or getattr(modified, "source_code", None)
            or ""
        )

        diff_parts.append(
            f"--- {normalize_path(path)}\n{patch}"
        )

    total_added = int(
        sum(
            int(getattr(m, "added_lines", 0) or 0)
            for m in commit.modified_files
            if normalize_path(
                m.new_path or m.old_path or ""
            )
            in relevant_files
        )
    )

    total_deleted = int(
        sum(
            int(getattr(m, "deleted_lines", 0) or 0)
            for m in commit.modified_files
            if normalize_path(
                m.new_path or m.old_path or ""
            )
            in relevant_files
        )
    )

    return {
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
        "ent": float(
            entropy_from_paths(
                commit,
                relevant_files,
            )
        ),
        "ndev": int(metrics["ndev"]),
        "age": float(metrics["age"]),
        "nuc": int(metrics["nuc"]),
        "aexp": int(metrics["aexp"]),
        "arexp": int(metrics["arexp"]),
        "asexp": int(metrics["asexp"]),
        "message": message,
        "diff": "\n\n".join(diff_parts),
    }


def recent_commit_window_for_target(
    repo_path: Path,
    extensions: Set[str],
    target_relevant: int,
) -> List[str]:

    repo = Repo(str(repo_path))

    newest_to_oldest = []
    relevant_found = 0

    for inspected, git_commit in enumerate(
        repo.iter_commits(),
        start=1,
    ):

        newest_to_oldest.append(
            git_commit.hexsha
        )

        try:
            changed_paths = list(
                git_commit.stats.files.keys()
            )
        except Exception:
            changed_paths = []

        if any(
            file_is_relevant(
                path,
                extensions,
            )
            for path in changed_paths
        ):
            relevant_found += 1

        if inspected % 500 == 0:
            print(
                f"Pre-scan: inspected {inspected} Git commits; "
                f"found {relevant_found}/{target_relevant} "
                f"relevant commits",
                flush=True,
            )

        if relevant_found >= target_relevant:
            break

    print(
        f"Pre-scan complete: {relevant_found} relevant commits found.",
        flush=True,
    )

    if relevant_found < target_relevant:
        print(
            f"WARNING: only {relevant_found} relevant commits "
            f"were found; requested {target_relevant}.",
            flush=True,
        )

    return list(reversed(newest_to_oldest))


def save_checkpoint(
    rows: List[Dict[str, object]],
    output_path: Path,
) -> None:

    df = pd.DataFrame(
        rows,
        columns=COLUMNS,
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Checkpoint saved: {len(df)} rows -> {output_path}",
        flush=True,
    )


def mine_krita(
    spec: RepoSpec,
    target_rows: int,
) -> pd.DataFrame:

    print("\n" + "=" * 72)
    print("MINING KRITA")
    print("=" * 72)

    checkpoint_path = (
        spec.path.parent
        / "krita_cpp_checkpoint.csv"
    )

    final_path = (
        spec.path.parent
        / "cpp_krita_dataset.csv"
    )

    # ---------------------------------------------------------
    # Resume from checkpoint if it already exists
    # ---------------------------------------------------------

    existing_rows = []

    if checkpoint_path.exists():

        print(
            f"Found checkpoint: {checkpoint_path}",
            flush=True,
        )

        checkpoint_df = pd.read_csv(
            checkpoint_path
        )

        existing_rows = checkpoint_df.to_dict(
            orient="records"
        )

        print(
            f"Loaded {len(existing_rows)} "
            f"checkpoint rows.",
            flush=True,
        )

        if len(existing_rows) >= target_rows:

            print(
                "Checkpoint already contains enough rows.",
                flush=True,
            )

            return checkpoint_df.tail(
                target_rows
            ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Pre-scan
    # ---------------------------------------------------------

    commit_hashes = recent_commit_window_for_target(
        spec.path,
        spec.extensions,
        target_rows,
    )

    print(
        f"Selected {len(commit_hashes)} Git commits "
        f"for detailed mining.",
        flush=True,
    )

    # ---------------------------------------------------------
    # Detailed mining
    # ---------------------------------------------------------

    history = FileHistoryTracker(
        spec.path
    )

    rows = existing_rows

    already_processed = {
        row["commit_id"]
        for row in rows
        if "commit_id" in row
    }

    total = len(commit_hashes)

    for index, commit in enumerate(
        Repository(
            str(spec.path),
            only_commits=commit_hashes,
        ).traverse_commits(),
        start=1,
    ):

        # Skip commits already stored in checkpoint
        if commit.hash in already_processed:

            continue

        row = commit_to_row(
            commit,
            spec,
            history,
        )

        changed_files = [
            normalize_path(
                m.new_path or m.old_path
            )
            for m in commit.modified_files
            if (m.new_path or m.old_path)
        ]

        commit_dt = (
            commit.author_date.replace(
                tzinfo=timezone.utc
            )
            if commit.author_date.tzinfo is None
            else commit.author_date.astimezone(
                timezone.utc
            )
        )

        history.record_commit(
            commit.author.name
            or commit.author.email
            or "unknown",
            commit_dt,
            changed_files,
        )

        if row is not None:
            rows.append(row)
            already_processed.add(
                commit.hash
            )

        # -----------------------------------------------------
        # Progress
        # -----------------------------------------------------

        if index % 100 == 0:

            print(
                f"Krita: processed {index}/{total} "
                f"Git commits; "
                f"retained {len(rows)}/{target_rows}",
                flush=True,
            )

        # -----------------------------------------------------
        # Checkpoint
        # -----------------------------------------------------

        if index % CHECKPOINT_EVERY == 0:

            save_checkpoint(
                rows,
                checkpoint_path,
            )

        if len(rows) >= target_rows:
            break

    # Final checkpoint
    save_checkpoint(
        rows,
        checkpoint_path,
    )

    df = pd.DataFrame(
        rows,
        columns=COLUMNS,
    )

    # Keep newest target rows
    if len(df) > target_rows:
        df = df.tail(
            target_rows
        ).reset_index(drop=True)

    # Save Krita-only dataset
    df.to_csv(
        final_path,
        index=False,
    )

    print("\n" + "=" * 72)
    print("KRITA COMPLETE")
    print("=" * 72)
    print(f"Rows: {len(df)}")
    print(f"Saved: {final_path}")
    print(f"Checkpoint: {checkpoint_path}")

    print("\nBug/fix labels:")
    print(
        df["buggy"].value_counts(
            dropna=False
        )
    )

    return df


def main():

    base_dir = Path(
        __file__
    ).resolve().parent

    krita = RepoSpec(
        name="krita",
        path=base_dir / "krita",
        project="cpp/krita",
        extensions=CPP_EXTS,
    )

    if not krita.path.exists():

        raise FileNotFoundError(
            f"Krita repository not found: "
            f"{krita.path}\n"
            f"Clone Krita beside this script first."
        )

    df = mine_krita(
        krita,
        TARGET_ROWS,
    )

    if len(df) < TARGET_ROWS:

        raise RuntimeError(
            f"Only {len(df)} Krita C++ rows "
            f"were produced; "
            f"{TARGET_ROWS} were required."
        )


if __name__ == "__main__":
    main()