"""
Multilingual SZZ Labeling Pipeline
==================================

Input:
    D:\embeddings_and_pca_code\data\python_cpp_23996_final.csv

Project column format:
    python/flask/...    -> language=python, repository=flask
    python/django/...   -> language=python, repository=django
    python/cpython/...  -> language=python, repository=cpython
    cpp/opencv/...      -> language=cpp, repository=opencv
    cpp/krita/...       -> language=cpp, repository=krita
    cpp/godot/...       -> language=cpp, repository=godot
    apache/...          -> language=java, repository=apache

The language column is created internally. It does NOT need to exist
in the input CSV.

SZZ:
    candidate fixing commit
        -> parent commit
        -> deleted/modified lines
        -> git blame deleted lines in parent
        -> defect-inducing commits
        -> buggy label

Output:
    D:\embeddings_and_pca_code\cpp_python_mining_code\python_cpp_23996_szz.csv
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import pandas as pd
import json
import re
import time
import sys


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"D:\embeddings_and_pca_code\cpp_python_mining_code")

INPUT_CSV = Path(
    r"D:\embeddings_and_pca_code\data\python_cpp_23996_final.csv"
)

OUTPUT_CSV = BASE_DIR / "python_cpp_23996_szz.csv"
CHECKPOINT_FILE = BASE_DIR / "szz_checkpoint.json"
RESULTS_FILE = BASE_DIR / "szz_results.json"


# ============================================================
# REPOSITORIES
# ============================================================

REPOSITORIES = {
    "flask": BASE_DIR / "flask",
    "django": BASE_DIR / "django",
    "cpython": BASE_DIR / "cpython",
    "opencv": BASE_DIR / "opencv",
    "krita": BASE_DIR / "krita",
    "godot": BASE_DIR / "godot",
}

# Language is determined from the FIRST component of project.
LANGUAGE_PREFIXES = {
    "python": "python",
    "cpp": "cpp",
    "apache": "java",
}


# ============================================================
# PERFORMANCE
# ============================================================

WORKERS = 4
CHECKPOINT_EVERY = 100
GIT_TIMEOUT = 120


# ============================================================
# GIT UTILITIES
# ============================================================

def run_git(repo, args, timeout=GIT_TIMEOUT):
    """Execute a git command inside a repository."""

    command = ["git", "-C", str(repo)] + args

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )

        if result.returncode != 0:
            return None

        return result.stdout

    except subprocess.TimeoutExpired:
        return None

    except Exception:
        return None


def is_valid_repository(repo):
    """Check whether a directory is a valid Git repository."""

    if not repo.exists():
        return False

    result = run_git(
        repo,
        ["rev-parse", "--is-inside-work-tree"]
    )

    return result is not None and result.strip() == "true"


# ============================================================
# PROJECT / LANGUAGE DETECTION
# ============================================================

def detect_language_and_repository(project):
    """
    Detect language and repository from the project column.

    Expected examples:
        python/flask       -> ("python", "flask")
        python/django      -> ("python", "django")
        python/cpython     -> ("python", "cpython")
        cpp/opencv         -> ("cpp", "opencv")
        cpp/krita          -> ("cpp", "krita")
        cpp/godot          -> ("cpp", "godot")
        apache/...         -> ("java", "apache")

    The first path component determines the language.
    The second path component determines the repository.
    """

    if pd.isna(project):
        return None, None

    project = str(project).strip().replace("\\", "/")

    # Remove leading/trailing slashes and empty components.
    parts = [
        part.strip()
        for part in project.split("/")
        if part.strip()
    ]

    if not parts:
        return None, None

    prefix = parts[0].lower()

    language = LANGUAGE_PREFIXES.get(prefix)

    if language is None:
        return None, None

    # For python/flask/... or cpp/opencv/..., second component
    # is the repository name.
    if len(parts) >= 2:
        repository = parts[1].lower()
    else:
        # For apache/... the repository is apache.
        repository = prefix

    return language, repository


# ============================================================
# DIFF PARSING
# ============================================================

def parse_deleted_line_ranges(diff_text):
    """
    Extract deleted-line ranges from a unified diff.

    The old/parent side of the diff is used because SZZ blames
    the lines in the parent commit.
    """

    ranges = []

    if not diff_text:
        return ranges

    current_file = None

    for line in diff_text.splitlines():

        # ----------------------------------------------------
        # File information
        # ----------------------------------------------------

        if line.startswith("diff --git "):

            match = re.match(
                r"diff --git a/(.*?) b/(.*)",
                line
            )

            if match:
                current_file = match.group(1)

            continue

        # ----------------------------------------------------
        # Ignore binary files
        # ----------------------------------------------------

        if line.startswith("Binary files"):
            current_file = None
            continue

        # ----------------------------------------------------
        # Diff hunk
        # ----------------------------------------------------

        if line.startswith("@@"):

            match = re.match(
                r"@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@",
                line
            )

            if not match:
                continue

            old_start = int(match.group(1))
            old_count = int(match.group(2) or 1)

            if current_file is None:
                continue

            # Deleted/modified lines are on the parent side.
            if old_count > 0:
                ranges.append(
                    (
                        current_file,
                        old_start,
                        old_start + old_count - 1,
                    )
                )

    return ranges


# ============================================================
# COMMIT DIFF
# ============================================================

def get_commit_diff(repo, commit):
    """Get a commit's first parent and diff against that parent."""

    parents = run_git(
        repo,
        ["rev-list", "--parents", "-n", "1", commit]
    )

    if not parents:
        return None

    parts = parents.strip().split()

    # Root commit has no parent.
    if len(parts) < 2:
        return None

    parent = parts[1]

    diff = run_git(
        repo,
        [
            "diff",
            "--unified=0",
            "--no-ext-diff",
            parent,
            commit,
        ],
    )

    if diff is None:
        return None

    return parent, diff


# ============================================================
# GIT BLAME
# ============================================================

def blame_range(repo, parent_commit, file_path, start_line, end_line):
    """Blame a specific line range in the parent commit."""

    if not file_path or file_path == "/dev/null":
        return []

    # Normalize possible diff prefixes.
    if file_path.startswith("a/"):
        file_path = file_path[2:]

    if file_path.startswith("b/"):
        file_path = file_path[2:]

    output = run_git(
        repo,
        [
            "blame",
            "--porcelain",
            "-L",
            f"{start_line},{end_line}",
            parent_commit,
            "--",
            file_path,
        ],
    )

    if not output:
        return []

    commits = []

    for line in output.splitlines():

        # Porcelain blame lines:
        # <40-character-sha> <orig-line> <final-line>
        match = re.match(
            r"^([0-9a-f]{40})\s+\d+\s+\d+",
            line,
        )

        if match:
            commits.append(match.group(1))

    return commits


# ============================================================
# PROCESS ONE FIXING COMMIT
# ============================================================

def process_commit(task):
    """
    Process one candidate fixing commit.

    task = (repository, commit)
    """

    repository, commit = task

    repo = REPOSITORIES.get(repository)

    if repo is None:
        return {
            "fix_commit": commit,
            "project": repository,
            "inducing_commits": [],
            "status": "missing_project",
        }

    if not is_valid_repository(repo):
        return {
            "fix_commit": commit,
            "project": repository,
            "inducing_commits": [],
            "status": "invalid_repository",
        }

    # --------------------------------------------------------
    # Get parent + diff
    # --------------------------------------------------------

    commit_data = get_commit_diff(repo, commit)

    if commit_data is None:
        return {
            "fix_commit": commit,
            "project": repository,
            "inducing_commits": [],
            "status": "no_parent_or_diff",
        }

    parent, diff = commit_data

    # --------------------------------------------------------
    # Find deleted/modified lines
    # --------------------------------------------------------

    deleted_ranges = parse_deleted_line_ranges(diff)

    if not deleted_ranges:
        return {
            "fix_commit": commit,
            "project": repository,
            "inducing_commits": [],
            "status": "no_deleted_lines",
        }

    inducing_commits = set()

    # --------------------------------------------------------
    # Blame deleted lines in parent
    # --------------------------------------------------------

    for file_path, start_line, end_line in deleted_ranges:

        blamed = blame_range(
            repo,
            parent,
            file_path,
            start_line,
            end_line,
        )

        for sha in blamed:

            # Ignore null SHA.
            if sha == "0" * 40:
                continue

            # Do not count fixing commit itself.
            if sha == commit:
                continue

            inducing_commits.add(sha)

    return {
        "fix_commit": commit,
        "project": repository,
        "inducing_commits": list(inducing_commits),
        "status": "success",
    }


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(results):
    """Save current SZZ results safely."""

    temp_file = CHECKPOINT_FILE.with_suffix(".tmp")

    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    temp_file.replace(CHECKPOINT_FILE)


def load_checkpoint():
    """Load previous checkpoint if it exists."""

    if not CHECKPOINT_FILE.exists():
        return {}

    try:
        with open(
            CHECKPOINT_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    except Exception:
        print("WARNING: Could not read checkpoint.")
        return {}


# ============================================================
# FINAL DATASET
# ============================================================

def finalize_dataset(df, results):

    print()
    print("=" * 70)
    print("GENERATING FINAL SZZ DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Build set of defect-inducing commits
    # --------------------------------------------------------

    inducing_commits = set()
    successful_results = 0

    for result in results.values():

        if result.get("status") != "success":
            continue

        successful_results += 1

        for commit in result.get(
            "inducing_commits",
            [],
        ):
            inducing_commits.add(commit)

    print(
        f"\nUnique SZZ defect-inducing commits: "
        f"{len(inducing_commits):,}"
    )

    print(
        f"Successful SZZ fix commits: "
        f"{successful_results:,}"
    )

    # --------------------------------------------------------
    # Generate final labels
    # --------------------------------------------------------

    output_df = df.copy()

    output_df["buggy"] = (
        output_df["commit_id"]
        .astype(str)
        .isin(inducing_commits)
    )

    output_df["label_source"] = "szz"

    # Preserve original candidate-fix information.
    output_df["candidate_fix"] = output_df["fix"]

    output_df["buggy"] = output_df["buggy"].astype(bool)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(
        f"\nFinal dataset saved to:\n"
        f"{OUTPUT_CSV}"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("\nFinal statistics:")

    print(
        f"Total rows: "
        f"{len(output_df):,}"
    )

    print(
        f"Buggy / defect-inducing: "
        f"{output_df['buggy'].sum():,}"
    )

    print(
        f"Clean: "
        f"{(~output_df['buggy']).sum():,}"
    )

    print(
        f"Buggy percentage: "
        f"{output_df['buggy'].mean() * 100:.2f}%"
    )

    print("\nBy language:")

    language_stats = (
        output_df
        .groupby("language")["buggy"]
        .agg(
            total="count",
            buggy="sum",
        )
    )

    language_stats["buggy_%"] = (
        language_stats["buggy"]
        / language_stats["total"]
        * 100
    )

    print(language_stats.to_string())

    print("\nBy project:")

    project_stats = (
        output_df
        .groupby("project")["buggy"]
        .agg(
            total="count",
            buggy="sum",
        )
    )

    project_stats["buggy_%"] = (
        project_stats["buggy"]
        / project_stats["total"]
        * 100
    )

    print(project_stats.to_string())

    print("\nDone.")


# ============================================================
# MAIN SZZ PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("MULTILINGUAL SZZ LABELING")
    print("=" * 70)

    start_time = time.time()

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not INPUT_CSV.exists():

        print(
            f"\nERROR: Input file not found:\n"
            f"{INPUT_CSV}"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_CSV)

    print(f"Total rows: {len(df):,}")

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    # IMPORTANT:
    # 'language' is NOT required because it is detected from
    # the project column.
    required_columns = [
        "commit_id",
        "project",
        "fix",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        print("\nERROR: Missing required columns:")

        for col in missing:
            print(f"  - {col}")

        sys.exit(1)

    # --------------------------------------------------------
    # Detect language + repository
    # --------------------------------------------------------

    print("\nDetecting language from project column...")

    detected = df["project"].apply(
        detect_language_and_repository
    )

    df["language"] = detected.apply(
        lambda x: x[0]
    )

    df["_repository"] = detected.apply(
        lambda x: x[1]
    )

    print("\nDetected languages:")

    print(
        df["language"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nDetected repositories:")

    print(
        df["_repository"]
        .value_counts(dropna=False)
        .to_string()
    )

    # --------------------------------------------------------
    # Warn about unsupported repositories
    # --------------------------------------------------------

    detected_repositories = set(
        df["_repository"].dropna().astype(str)
    )

    unsupported = sorted(
        detected_repositories
        - set(REPOSITORIES.keys())
    )

    if unsupported:

        print(
            "\nWARNING: These repositories are present in the "
            "dataset but are not cloned/configured:"
        )

        for repo in unsupported:
            print(f"  - {repo}")

        print(
            "\nThey will be skipped during SZZ."
        )

    # --------------------------------------------------------
    # Repository validation
    # --------------------------------------------------------

    print("\nChecking repositories...")

    for repository, repo_path in REPOSITORIES.items():

        status = is_valid_repository(repo_path)

        print(
            f"{repository:<12} "
            f"{'OK' if status else 'NOT FOUND'} "
            f"{repo_path}"
        )

    # --------------------------------------------------------
    # Project statistics
    # --------------------------------------------------------

    print("\nProjects:")

    print(
        df["project"]
        .value_counts()
        .to_string()
    )

    print("\nLanguages:")

    print(
        df["language"]
        .value_counts(dropna=False)
        .to_string()
    )

    # --------------------------------------------------------
    # Candidate fixing commits
    # --------------------------------------------------------

    print(
        "\nThe existing 'fix' column is used ONLY to identify "
        "candidate fixing commits."
    )

    print(
        "SZZ determines the actual defect-inducing commits."
    )

    fix_mask = (
        df["fix"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin([
            "true",
            "1",
            "yes",
        ])
    )

    fix_df = df[fix_mask].copy()

    print(
        f"\nCandidate fixing commits: "
        f"{len(fix_df):,}"
    )

    if len(fix_df) == 0:

        print(
            "\nERROR: No candidate fixing commits found."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Create SZZ tasks
    # --------------------------------------------------------

    tasks = []
    skipped_invalid = 0

    for _, row in fix_df.iterrows():

        repository = row["_repository"]
        commit = str(row["commit_id"]).strip()

        # Unknown project format.
        if pd.isna(repository):
            skipped_invalid += 1
            continue

        repository = str(repository).strip().lower()

        # Repository not configured/cloned.
        if repository not in REPOSITORIES:
            skipped_invalid += 1
            continue

        # Commit must be a full SHA.
        if not re.fullmatch(
            r"[0-9a-fA-F]{40}",
            commit,
        ):
            skipped_invalid += 1
            continue

        tasks.append(
            (
                repository,
                commit.lower(),
            )
        )

    # Remove duplicate repository/commit tasks.
    tasks = list(dict.fromkeys(tasks))

    print(
        f"Valid SZZ tasks: {len(tasks):,}"
    )

    if skipped_invalid:
        print(
            f"Skipped invalid/unsupported tasks: "
            f"{skipped_invalid:,}"
        )

    if not tasks:

        print(
            "\nERROR: No valid SZZ tasks found."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    results = load_checkpoint()

    completed = set(results.keys())

    remaining_tasks = [
        task
        for task in tasks
        if f"{task[0]}::{task[1]}" not in completed
    ]

    print(
        f"Already completed: {len(completed):,}"
    )

    print(
        f"Remaining: {len(remaining_tasks):,}"
    )

    # --------------------------------------------------------
    # Nothing left to process
    # --------------------------------------------------------

    if not remaining_tasks:

        print(
            "\nAll SZZ tasks already completed."
        )

        finalize_dataset(
            df.drop(columns=["_repository"]),
            results,
        )

        return

    # --------------------------------------------------------
    # Multiprocessing
    # --------------------------------------------------------

    print(
        f"\nStarting SZZ with "
        f"{WORKERS} workers..."
    )

    print(
        "NOTE: GPU is not used during SZZ. "
        "This stage is Git/CPU/SSD bound."
    )

    processed = 0
    successful = 0
    failed = 0
    total_tasks = len(remaining_tasks)

    with ProcessPoolExecutor(
        max_workers=WORKERS
    ) as executor:

        futures = {
            executor.submit(
                process_commit,
                task,
            ): task
            for task in remaining_tasks
        }

        for future in as_completed(futures):

            task = futures[future]
            key = f"{task[0]}::{task[1]}"

            try:

                result = future.result()

                results[key] = result

                if result["status"] == "success":
                    successful += 1
                else:
                    failed += 1

            except Exception as e:

                failed += 1

                results[key] = {
                    "fix_commit": task[1],
                    "project": task[0],
                    "inducing_commits": [],
                    "status": "exception",
                    "error": str(e),
                }

            processed += 1

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            elapsed = time.time() - start_time

            rate = (
                processed / elapsed
                if elapsed > 0
                else 0
            )

            remaining = total_tasks - processed

            eta = (
                remaining / rate
                if rate > 0
                else 0
            )

            print(
                f"\rProgress: "
                f"{processed:,}/{total_tasks:,} "
                f"({processed / total_tasks * 100:.1f}%) | "
                f"Success: {successful:,} | "
                f"Failed: {failed:,} | "
                f"Rate: {rate:.2f}/s | "
                f"ETA: {eta / 60:.1f} min",
                end="",
                flush=True,
            )

            # ------------------------------------------------
            # Checkpoint
            # ------------------------------------------------

            if processed % CHECKPOINT_EVERY == 0:

                save_checkpoint(results)

                print(
                    "\nCheckpoint saved."
                )

    print("\n")

    # --------------------------------------------------------
    # Final checkpoint
    # --------------------------------------------------------

    save_checkpoint(results)

    # --------------------------------------------------------
    # Save raw results
    # --------------------------------------------------------

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        f"Raw SZZ results saved to:\n"
        f"{RESULTS_FILE}"
    )

    # --------------------------------------------------------
    # Generate final dataset
    # --------------------------------------------------------

    finalize_dataset(
        df.drop(columns=["_repository"]),
        results,
    )

    elapsed = time.time() - start_time

    print(
        f"\nTotal runtime: "
        f"{elapsed / 3600:.2f} hours"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
