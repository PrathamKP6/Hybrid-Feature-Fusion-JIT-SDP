import os
import re
import pickle
import math
from collections import defaultdict, Counter
from datetime import datetime

import pandas as pd
from git import Repo
from pydriller import Repository


# ============================================================
# CONFIGURATION
# ============================================================

CPYTHON_PATH = r"D:\embeddings_and_pca_code\cpp_python_mining_code\cpython"

DJANGO_CSV = "python_django_dataset.csv"
FLASK_CSV = "python_flask_dataset.csv"

CPYTHON_CSV = "python_cpython_dataset.csv"
FINAL_CSV = "python_final_12k.csv"

CHECKPOINT_CSV = "cpython_python_checkpoint.csv"
HISTORY_CHECKPOINT = "cpython_python_history.pkl"

TARGET_TOTAL = 12000

PYTHON_EXTENSIONS = {".py"}

BUG_KEYWORDS = (
    "fix",
    "bug",
    "error",
    "issue",
    "patch",
    "resolve",
    "hotfix",
)

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


# ============================================================
# HISTORY TRACKER
# ============================================================

class FileHistoryTracker:

    def __init__(self):
        # file -> previous commit dates
        self.file_commit_dates = defaultdict(list)

        # file -> developers
        self.file_developers = defaultdict(set)

        # file -> number of commits
        self.file_commit_count = defaultdict(int)

        # author -> number of commits
        self.author_commit_count = Counter()

        # (author, subsystem) -> number of commits
        self.author_subsystem_count = Counter()

        # file -> recent commits
        self.recent_commits = defaultdict(list)

    def compute_metrics(
        self,
        relevant_files,
        author,
        commit_date,
        subsystems,
    ):
        # ----------------------------------------------------
        # Number of developers
        # ----------------------------------------------------
        developers = set()

        for f in relevant_files:
            developers.update(self.file_developers.get(f, set()))

        ndev = len(developers)

        # ----------------------------------------------------
        # Age
        # ----------------------------------------------------
        ages = []

        for f in relevant_files:
            dates = self.file_commit_dates.get(f, [])

            if dates:
                last_date = max(dates)

                try:
                    age_days = (
                        commit_date - last_date
                    ).total_seconds() / 86400.0

                    ages.append(max(age_days, 0.0))

                except Exception:
                    pass

        age = sum(ages) / len(ages) if ages else 0.0

        # ----------------------------------------------------
        # Number of previous commits
        # ----------------------------------------------------
        nuc = sum(
            self.file_commit_count.get(f, 0)
            for f in relevant_files
        )

        # ----------------------------------------------------
        # Author experience
        # ----------------------------------------------------
        aexp = self.author_commit_count.get(author, 0)

        # ----------------------------------------------------
        # Recent author experience
        # ----------------------------------------------------
        arexp = sum(
            1
            for f in relevant_files
            for previous_author in self.file_developers.get(f, set())
            if previous_author == author
        )

        # ----------------------------------------------------
        # Subsystem experience
        # ----------------------------------------------------
        as_experience = 0

        for subsystem in subsystems:
            as_experience += self.author_subsystem_count.get(
                (author, subsystem),
                0,
            )

        asexp = as_experience

        return {
            "ndev": ndev,
            "age": age,
            "nuc": nuc,
            "aexp": aexp,
            "arexp": arexp,
            "asexp": asexp,
        }

    def update(
        self,
        relevant_files,
        author,
        commit_date,
        subsystems,
        commit_hash,
    ):
        # ----------------------------------------------------
        # Update file histories
        # ----------------------------------------------------
        for f in relevant_files:

            self.file_commit_dates[f].append(commit_date)

            self.file_developers[f].add(author)

            self.file_commit_count[f] += 1

            self.recent_commits[f].append(commit_hash)

        # ----------------------------------------------------
        # Author history
        # ----------------------------------------------------
        self.author_commit_count[author] += 1

        # ----------------------------------------------------
        # Subsystem history
        # ----------------------------------------------------
        for subsystem in subsystems:
            self.author_subsystem_count[
                (author, subsystem)
            ] += 1


# ============================================================
# BASIC HELPERS
# ============================================================

def is_python_file(path):
    if not path:
        return False

    _, ext = os.path.splitext(path.lower())

    return ext in PYTHON_EXTENSIONS


def commit_is_buggy(message):
    if not message:
        return False

    message_lower = message.lower()

    return any(
        keyword in message_lower
        for keyword in BUG_KEYWORDS
    )


def get_file_path(modification):
    """
    PyDriller exposes old_path/new_path.
    For renamed/deleted/added files, select the path that exists.
    """

    new_path = getattr(modification, "new_path", None)
    old_path = getattr(modification, "old_path", None)

    if new_path:
        return new_path

    return old_path


def touched_python_files(commit):
    """
    Return ONLY Python files touched by this commit.

    A commit is considered relevant if at least one
    touched file ends with .py.
    """

    files = []

    for modification in commit.modified_files:

        path = get_file_path(modification)

        if is_python_file(path):
            files.append(path)

    return files


def get_subsystems(files):
    """
    Approximate subsystem/directory using the first
    directory component.
    """

    subsystems = set()

    for f in files:

        parts = f.replace("\\", "/").split("/")

        if len(parts) > 1:
            subsystems.add(parts[0])
        else:
            subsystems.add("root")

    return subsystems


# ============================================================
# ENTROPY
# ============================================================

def calculate_entropy(files):
    if not files:
        return 0.0

    counts = Counter(files)

    total = sum(counts.values())

    if total == 0:
        return 0.0

    entropy = 0.0

    for count in counts.values():

        p = count / total

        entropy -= p * math.log2(p)

    return entropy


# ============================================================
# COMMIT ROW
# ============================================================

def commit_to_row(
    commit,
    history,
):
    relevant_files = touched_python_files(commit)

    # --------------------------------------------------------
    # IMPORTANT:
    # Commit is ignored unless it touches Python files.
    # --------------------------------------------------------
    if not relevant_files:
        return None

    author = (
        commit.author.email
        if commit.author
        else "unknown"
    )

    commit_date = commit.author_date

    subsystems = get_subsystems(relevant_files)

    # --------------------------------------------------------
    # History metrics MUST be calculated BEFORE updating
    # history with current commit.
    # --------------------------------------------------------
    metrics = history.compute_metrics(
        relevant_files=relevant_files,
        author=author,
        commit_date=commit_date,
        subsystems=subsystems,
    )

    # --------------------------------------------------------
    # LOC metrics
    # --------------------------------------------------------
    la = 0
    ld = 0

    # --------------------------------------------------------
    # File / directory / subsystem metrics
    # --------------------------------------------------------
    touched_paths = []

    for modification in commit.modified_files:

        path = get_file_path(modification)

        if path:
            touched_paths.append(path)

        if path in relevant_files:

            added = getattr(
                modification,
                "added_lines",
                0,
            ) or 0

            deleted = getattr(
                modification,
                "deleted_lines",
                0,
            ) or 0

            la += added
            ld += deleted

    # --------------------------------------------------------
    # Number of files
    # --------------------------------------------------------
    nf = len(set(relevant_files))

    # --------------------------------------------------------
    # Number of directories
    # --------------------------------------------------------
    directories = set()

    for f in relevant_files:

        directory = os.path.dirname(
            f.replace("\\", "/")
        )

        directories.add(
            directory if directory else "root"
        )

    nd = len(directories)

    # --------------------------------------------------------
    # Number of subsystems
    # --------------------------------------------------------
    ns = len(subsystems)

    # --------------------------------------------------------
    # Entropy
    # --------------------------------------------------------
    ent = calculate_entropy(relevant_files)

    # --------------------------------------------------------
    # Bug/fix label
    # --------------------------------------------------------
    buggy = int(
        commit_is_buggy(commit.msg)
    )

    fix = buggy

    # --------------------------------------------------------
    # Diff
    # --------------------------------------------------------
    diff_text = ""

    try:
        diff_text = commit.diff
    except Exception:
        diff_text = ""

    # --------------------------------------------------------
    # Create row
    # --------------------------------------------------------
    row = {
        "commit_id": commit.hash,
        "project": "cpython",
        "buggy": buggy,
        "fix": fix,
        "year": commit_date.year,
        "author_date": commit_date.isoformat(),
        "la": la,
        "ld": ld,
        "nf": nf,
        "nd": nd,
        "ns": ns,
        "ent": ent,
        "ndev": metrics["ndev"],
        "age": metrics["age"],
        "nuc": metrics["nuc"],
        "aexp": metrics["aexp"],
        "arexp": metrics["arexp"],
        "asexp": metrics["asexp"],
        "message": commit.msg,
        "diff": diff_text,
    }

    # --------------------------------------------------------
    # Update history AFTER creating row.
    # --------------------------------------------------------
    history.update(
        relevant_files=relevant_files,
        author=author,
        commit_date=commit_date,
        subsystems=subsystems,
        commit_hash=commit.hash,
    )

    return row


# ============================================================
# SERIALIZATION
# ============================================================

def save_history(history):

    with open(
        HISTORY_CHECKPOINT,
        "wb",
    ) as f:

        pickle.dump(history, f)


def load_history():

    if not os.path.exists(
        HISTORY_CHECKPOINT
    ):
        return FileHistoryTracker()

    try:

        with open(
            HISTORY_CHECKPOINT,
            "rb",
        ) as f:

            history = pickle.load(f)

        print(
            f"[RESUME] Loaded history checkpoint: "
            f"{HISTORY_CHECKPOINT}"
        )

        return history

    except Exception as e:

        print(
            "[WARNING] Could not load history checkpoint:"
        )

        print(e)

        return FileHistoryTracker()


# ============================================================
# ROW CHECKPOINT
# ============================================================

def save_rows(rows):

    if not rows:
        return

    df = pd.DataFrame(rows)

    df = df[COLUMNS]

    df.to_csv(
        CHECKPOINT_CSV,
        index=False,
    )


def load_rows():

    if not os.path.exists(
        CHECKPOINT_CSV
    ):
        return []

    try:

        df = pd.read_csv(
            CHECKPOINT_CSV
        )

        # Check checkpoint schema.
        checkpoint_columns = list(
            df.columns
        )

        if set(checkpoint_columns) != set(COLUMNS):

            print(
                "[WARNING] Existing checkpoint "
                "has unexpected columns."
            )

            print(
                "Expected:",
                COLUMNS,
            )

            print(
                "Found:",
                checkpoint_columns,
            )

            return []

        df = df[COLUMNS]

        print(
            f"[RESUME] Loaded "
            f"{len(df)} checkpoint rows."
        )

        return df.to_dict(
            orient="records"
        )

    except Exception as e:

        print(
            "[WARNING] Could not load row checkpoint:"
        )

        print(e)

        return []


# ============================================================
# FIRST PASS:
# FIND RELEVANT CPYTHON COMMITS
# ============================================================

def collect_python_commits(
    repo_path,
    already_processed,
    target_rows,
):
    """
    Scan CPython history chronologically.

    ONLY commits touching .py files are collected.
    """

    repo = Repo(repo_path)

    print(
        "\n[SCAN] Finding CPython commits "
        "that touch Python files..."
    )

    commit_hashes = []

    # --------------------------------------------------------
    # Git history newest -> oldest
    # --------------------------------------------------------
    commits = list(
        repo.iter_commits(
            "--all"
        )
    )

    # Chronological order
    commits.reverse()

    for commit in commits:

        commit_hash = commit.hexsha

        # ----------------------------------------------------
        # Resume:
        # already processed commits are skipped.
        # ----------------------------------------------------
        if commit_hash in already_processed:
            continue

        # ----------------------------------------------------
        # Inspect changed files directly with GitPython.
        # ----------------------------------------------------
        touches_python = False

        try:

            if commit.parents:

                parent = commit.parents[0]

                diffs = parent.diff(
                    commit
                )

                for diff in diffs:

                    old_path = diff.a_path
                    new_path = diff.b_path

                    if (
                        is_python_file(old_path)
                        or
                        is_python_file(new_path)
                    ):
                        touches_python = True
                        break

            else:
                # Initial commit
                for path in commit.stats.files:

                    if is_python_file(path):
                        touches_python = True
                        break

        except Exception:

            # Fall back to commit stats.
            for path in commit.stats.files:

                if is_python_file(path):
                    touches_python = True
                    break

        # ----------------------------------------------------
        # ONLY Python commits are selected.
        # ----------------------------------------------------
        if touches_python:

            commit_hashes.append(
                commit_hash
            )

            if len(commit_hashes) >= target_rows:
                break

    print(
        f"[SCAN] Found "
        f"{len(commit_hashes)} new Python commits."
    )

    return commit_hashes


# ============================================================
# MINE CPYTHON
# ============================================================

def mine_cpython(target_rows):

    print("\n" + "=" * 70)
    print("CPYTHON MINING")
    print("=" * 70)

    if not os.path.exists(
        CPYTHON_PATH
    ):

        raise FileNotFoundError(
            f"CPython repository not found:\n"
            f"{CPYTHON_PATH}"
        )

    # --------------------------------------------------------
    # Load previous rows
    # --------------------------------------------------------
    rows = load_rows()

    # --------------------------------------------------------
    # If checkpoint already has enough rows,
    # no additional mining required.
    # --------------------------------------------------------
    if len(rows) >= target_rows:

        print(
            f"[RESUME] CPython checkpoint already "
            f"contains {len(rows)} rows."
        )

        rows = rows[:target_rows]

        pd.DataFrame(
            rows,
            columns=COLUMNS,
        ).to_csv(
            CPYTHON_CSV,
            index=False,
        )

        return rows

    # --------------------------------------------------------
    # Load history.
    # --------------------------------------------------------
    history = load_history()

    already_processed = {
        row["commit_id"]
        for row in rows
        if row.get("commit_id")
    }

    remaining = (
        target_rows - len(rows)
    )

    print(
        f"[INFO] Target CPython rows : {target_rows}"
    )

    print(
        f"[INFO] Existing CPython rows: {len(rows)}"
    )

    print(
        f"[INFO] Remaining rows       : {remaining}"
    )

    # --------------------------------------------------------
    # FIRST PASS
    # --------------------------------------------------------
    commit_hashes = collect_python_commits(
        repo_path=CPYTHON_PATH,
        already_processed=already_processed,
        target_rows=remaining,
    )

    if not commit_hashes:

        print(
            "[WARNING] No new Python commits found."
        )

        return rows

    # --------------------------------------------------------
    # SECOND PASS:
    # PyDriller detailed processing.
    # --------------------------------------------------------
    print(
        "\n[MINING] Processing selected "
        "Python commits with PyDriller..."
    )

    processed_count = 0

    for commit in Repository(
        CPYTHON_PATH,
        only_commits=commit_hashes,
    ).traverse_commits():

        if len(rows) >= target_rows:
            break

        try:

            row = commit_to_row(
                commit,
                history,
            )

            # ------------------------------------------------
            # Safety check:
            # commit_to_row returns None if no Python files.
            # ------------------------------------------------
            if row is None:
                continue

            rows.append(row)

            processed_count += 1

            # ------------------------------------------------
            # Checkpoint every 100 rows.
            # ------------------------------------------------
            if (
                processed_count % 100 == 0
            ):

                save_rows(rows)
                save_history(history)

                print(
                    f"[CHECKPOINT] "
                    f"CPython rows: "
                    f"{len(rows)}/{target_rows}"
                )

        except Exception as e:

            print(
                f"[WARNING] Failed commit "
                f"{commit.hash}: {e}"
            )

    # --------------------------------------------------------
    # Final checkpoint.
    # --------------------------------------------------------
    save_rows(rows)
    save_history(history)

    # --------------------------------------------------------
    # Trim to requested CPython count.
    # --------------------------------------------------------
    rows = rows[:target_rows]

    df = pd.DataFrame(
        rows,
        columns=COLUMNS,
    )

    df.to_csv(
        CPYTHON_CSV,
        index=False,
    )

    print(
        f"\n[DONE] CPython dataset written:"
    )

    print(
        f"       {CPYTHON_CSV}"
    )

    print(
        f"[DONE] CPython rows: {len(df)}"
    )

    return rows


# ============================================================
# COLUMN SCHEMA COMPARISON
# ============================================================

def compare_columns(
    csv_files
):
    """
    Compare the column names of all three datasets.

    Reports:
      - exact column names
      - missing columns
      - extra columns
      - ordering differences

    Returns True only if all datasets have exactly
    the same set of columns.
    """

    print("\n" + "=" * 70)
    print("COLUMN SCHEMA CHECK")
    print("=" * 70)

    schemas = {}

    # --------------------------------------------------------
    # Load columns only
    # --------------------------------------------------------
    for csv_file in csv_files:

        if not os.path.exists(csv_file):

            print(
                f"\n[ERROR] File not found: "
                f"{csv_file}"
            )

            return False

        df = pd.read_csv(
            csv_file,
            nrows=0,
        )

        schemas[csv_file] = list(
            df.columns
        )

        print(
            f"\n{csv_file}"
        )

        print(
            f"Columns ({len(df.columns)}):"
        )

        for i, column in enumerate(
            df.columns,
            start=1,
        ):

            print(
                f"  {i}. {column}"
            )

    # --------------------------------------------------------
    # Reference schema = Django
    # --------------------------------------------------------
    reference_file = csv_files[0]

    reference_columns = schemas[
        reference_file
    ]

    reference_set = set(
        reference_columns
    )

    anomalies_found = False

    # --------------------------------------------------------
    # Compare every dataset against Django.
    # --------------------------------------------------------
    for csv_file in csv_files[1:]:

        columns = schemas[csv_file]

        column_set = set(columns)

        missing = (
            reference_set - column_set
        )

        extra = (
            column_set - reference_set
        )

        order_same = (
            columns == reference_columns
        )

        # ----------------------------------------------------
        # Missing
        # ----------------------------------------------------
        if missing:

            anomalies_found = True

            print(
                f"\n[ANOMALY] {csv_file}"
            )

            print(
                "  Missing columns:"
            )

            for col in sorted(
                missing
            ):

                print(
                    f"    - {col}"
                )

        # ----------------------------------------------------
        # Extra
        # ----------------------------------------------------
        if extra:

            anomalies_found = True

            print(
                f"\n[ANOMALY] {csv_file}"
            )

            print(
                "  Extra columns:"
            )

            for col in sorted(
                extra
            ):

                print(
                    f"    + {col}"
                )

        # ----------------------------------------------------
        # Same names but different order
        # ----------------------------------------------------
        if (
            not missing
            and not extra
            and not order_same
        ):

            print(
                f"\n[INFO] {csv_file}"
            )

            print(
                "  Column names match, "
                "but column order differs."
            )

            print(
                "  This is NOT a schema anomaly."
            )

    # --------------------------------------------------------
    # Compare against expected schema as well.
    # --------------------------------------------------------
    expected_set = set(COLUMNS)

    for csv_file, columns in schemas.items():

        column_set = set(columns)

        missing_expected = (
            expected_set - column_set
        )

        extra_expected = (
            column_set - expected_set
        )

        if missing_expected:

            anomalies_found = True

            print(
                f"\n[ANOMALY] {csv_file}"
            )

            print(
                "  Missing expected columns:"
            )

            for col in sorted(
                missing_expected
            ):

                print(
                    f"    - {col}"
                )

        if extra_expected:

            anomalies_found = True

            print(
                f"\n[ANOMALY] {csv_file}"
            )

            print(
                "  Unexpected columns:"
            )

            for col in sorted(
                extra_expected
            ):

                print(
                    f"    + {col}"
                )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------
    print("\n" + "-" * 70)

    if anomalies_found:

        print(
            "[RESULT] COLUMN ANOMALIES FOUND."
        )

        print(
            "[RESULT] Merge will NOT be performed."
        )

        return False

    print(
        "[RESULT] All three CSV files have "
        "the same column names."
    )

    print(
        "[RESULT] Column order will be normalized "
        "before merging."
    )

    return True


# ============================================================
# FINAL MERGE
# ============================================================

def combine_python_datasets():

    print("\n" + "=" * 70)
    print("FINAL DATASET MERGE")
    print("=" * 70)

    csv_files = [
        DJANGO_CSV,
        FLASK_CSV,
        CPYTHON_CSV,
    ]

    # --------------------------------------------------------
    # IMPORTANT:
    # Check schemas BEFORE reading/merging datasets.
    # --------------------------------------------------------
    schema_ok = compare_columns(
        csv_files
    )

    if not schema_ok:

        print(
            "\n[STOP] Fix the column anomalies "
            "before merging."
        )

        return

    # --------------------------------------------------------
    # Read datasets
    # --------------------------------------------------------
    datasets = []

    for csv_file in csv_files:

        df = pd.read_csv(
            csv_file
        )

        # Normalize column order.
        df = df[COLUMNS]

        datasets.append(df)

        print(
            f"[LOAD] {csv_file}: "
            f"{len(df)} rows"
        )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------
    final_df = pd.concat(
        datasets,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Final maximum = 12,000
    # --------------------------------------------------------
    if len(final_df) > TARGET_TOTAL:

        final_df = final_df.head(
            TARGET_TOTAL
        )

    # --------------------------------------------------------
    # Final schema check
    # --------------------------------------------------------
    if list(final_df.columns) != COLUMNS:

        raise RuntimeError(
            "Final dataset schema does not "
            "match expected COLUMNS."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    final_df.to_csv(
        FINAL_CSV,
        index=False,
    )

    print(
        f"\n[DONE] Final dataset:"
    )

    print(
        f"       {FINAL_CSV}"
    )

    print(
        f"[DONE] Final rows: "
        f"{len(final_df)}"
    )

    print(
        "\nProject distribution:"
    )

    print(
        final_df["project"]
        .value_counts()
        .to_string()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CPYTHON PYTHON DATASET MINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Check Django and Flask datasets
    # --------------------------------------------------------
    if not os.path.exists(
        DJANGO_CSV
    ):

        raise FileNotFoundError(
            f"Missing Django dataset:\n"
            f"{DJANGO_CSV}"
        )

    if not os.path.exists(
        FLASK_CSV
    ):

        raise FileNotFoundError(
            f"Missing Flask dataset:\n"
            f"{FLASK_CSV}"
        )

    # --------------------------------------------------------
    # Count existing rows
    # --------------------------------------------------------
    django_df = pd.read_csv(
        DJANGO_CSV
    )

    flask_df = pd.read_csv(
        FLASK_CSV
    )

    django_rows = len(
        django_df
    )

    flask_rows = len(
        flask_df
    )

    cpython_target = (
        TARGET_TOTAL
        - django_rows
        - flask_rows
    )

    print(
        f"\nDjango rows : {django_rows}"
    )

    print(
        f"Flask rows  : {flask_rows}"
    )

    print(
        f"Target total: {TARGET_TOTAL}"
    )

    print(
        f"CPython required: "
        f"{cpython_target}"
    )

    # --------------------------------------------------------
    # If Django + Flask already exceed 12k
    # --------------------------------------------------------
    if cpython_target < 0:

        raise RuntimeError(
            "Django + Flask datasets already "
            "contain more than 12,000 rows."
        )

    # --------------------------------------------------------
    # Mine CPython
    # --------------------------------------------------------
    if cpython_target > 0:

        mine_cpython(
            cpython_target
        )

    else:

        print(
            "\n[INFO] No CPython rows required."
        )

        # Create empty CPython CSV with correct schema.
        pd.DataFrame(
            columns=COLUMNS
        ).to_csv(
            CPYTHON_CSV,
            index=False,
        )

    # --------------------------------------------------------
    # Compare columns and merge
    # --------------------------------------------------------
    combine_python_datasets()

    print(
        "\n" + "=" * 70
    )

    print(
        "PROCESS COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()