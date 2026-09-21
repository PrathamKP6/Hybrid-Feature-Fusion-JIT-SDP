"""
Optimized, resumable and failure-aware Multilingual SZZ Pipeline
================================================================

Input:
    D:\embeddings_and_pca_code\data\python_cpp_23996_final.csv

Repositories:
    D:\embeddings_and_pca_code\cpp_python_mining_code\{repo}

Existing checkpoint is PRESERVED and reused.

Important design decisions
---------------------------
1. `no_deleted_lines` is NOT treated as a failure. It is a valid SZZ
   outcome: the candidate fix has no parent-side deleted/modified lines.
2. `no_parent_or_diff` is rechecked because the old script could not
   distinguish a real root commit from a Git command failure.
3. Real Git failures are retried.
4. Blame is first attempted once per file with multiple -L ranges.
5. If that fails, each range is retried individually.
6. A task that remains unresolved is explicitly recorded as `unresolved`;
   its incomplete blame is NOT used to create buggy labels.
7. Each worker process handles ONE task only. The parent process can kill
   a worker that exceeds TASK_TIMEOUT. Therefore one pathological commit
   cannot hang the whole run.
8. Checkpoint is saved after every completed task and at regular intervals.
9. The old checkpoint's successful results are retained.
10. Old ambiguous/error results are reprocessed.
"""

from pathlib import Path
import multiprocessing as mp
import subprocess
import pandas as pd
import json
import re
import time
import sys
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"D:\embeddings_and_pca_code\cpp_python_mining_code")
INPUT_CSV = Path(r"D:\embeddings_and_pca_code\data\python_cpp_23996_final.csv")

OUTPUT_CSV = BASE_DIR / "python_cpp_23996_szz.csv"
CHECKPOINT_FILE = BASE_DIR / "szz_checkpoint.json"
RESULTS_FILE = BASE_DIR / "szz_results.json"
FAILURE_REPORT = BASE_DIR / "szz_failure_report.json"

REPOSITORIES = {
    "flask": BASE_DIR / "flask",
    "django": BASE_DIR / "django",
    "cpython": BASE_DIR / "cpython",
    "opencv": BASE_DIR / "opencv",
    "krita": BASE_DIR / "krita",
    "godot": BASE_DIR / "godot",
}

LANGUAGE_PREFIXES = {
    "python": "python",
    "cpp": "cpp",
    "apache": "java",
}

# Number of simultaneously running task processes.
WORKERS = 6

# Timeout for one Git command.
GIT_TIMEOUT = 30

# HARD timeout for the complete processing of ONE fixing commit.
# This is what prevents a pathological commit from hanging the run.
TASK_TIMEOUT = 180

# Retry count for transient Git failures.
GIT_RETRIES = 2

# Save after every task. With only 87 remaining this is cheap and safest.
CHECKPOINT_EVERY = 1


# ============================================================
# GIT
# ============================================================

def run_git(repo, args, timeout=GIT_TIMEOUT):
    command = ["git", "-C", str(repo)] + args

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout
        )

        return {
            "ok": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr.strip(),
            "timeout": False,
            "returncode": result.returncode,
        }

    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"Git command timed out after {timeout}s",
            "timeout": True,
            "returncode": None,
        }

    except Exception as exc:
        return {
            "ok": False,
            "stdout": "",
            "stderr": str(exc),
            "timeout": False,
            "returncode": None,
        }


def run_git_retry(repo, args):
    last = None

    for attempt in range(GIT_RETRIES + 1):
        last = run_git(repo, args)

        if last["ok"]:
            return last

        # A normal Git error is retried too because repository/index
        # state or Windows file contention can occasionally be transient.
        if attempt < GIT_RETRIES:
            time.sleep(0.5 * (attempt + 1))

    return last


def is_valid_repository(repo):
    result = run_git_retry(
        repo,
        ["rev-parse", "--is-inside-work-tree"]
    )
    return result["ok"] and result["stdout"].strip() == "true"


# ============================================================
# PROJECT DETECTION
# ============================================================

def detect_language_and_repository(project):
    if pd.isna(project):
        return None, None

    parts = [
        p.strip()
        for p in str(project).strip().replace("\\", "/").split("/")
        if p.strip()
    ]

    if not parts:
        return None, None

    language = LANGUAGE_PREFIXES.get(parts[0].lower())
    repository = parts[1].lower() if len(parts) >= 2 else None

    return language, repository


# ============================================================
# DIFF PARSING
# ============================================================

def parse_deleted_line_ranges(diff_text):
    ranges = []
    current_file = None

    if not diff_text:
        return ranges

    for line in diff_text.splitlines():

        if line.startswith("diff --git "):

            match = re.match(
                r"diff --git a/(.*?) b/(.*)",
                line
            )

            current_file = match.group(1) if match else None
            continue

        if line.startswith("Binary files"):
            current_file = None
            continue

        if line.startswith("@@"):

            match = re.match(
                r"@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@",
                line
            )

            if not match or current_file is None:
                continue

            old_start = int(match.group(1))
            old_count = int(match.group(2) or 1)

            if old_count > 0:
                ranges.append(
                    (
                        current_file,
                        old_start,
                        old_start + old_count - 1
                    )
                )

    return ranges


# ============================================================
# COMMIT DIFF
# ============================================================

def get_commit_diff(repo, commit):

    result = run_git_retry(
        repo,
        [
            "rev-list",
            "--parents",
            "-n",
            "1",
            commit
        ]
    )

    if not result["ok"]:

        return {
            "status": (
                "git_timeout"
                if result["timeout"]
                else "git_error"
            ),
            "error": result["stderr"]
        }

    parts = result["stdout"].strip().split()

    if len(parts) < 2:

        return {
            "status": "no_parent",
            "error": "Commit has no parent (root commit or invalid history)"
        }

    parent = parts[1]

    result = run_git_retry(
        repo,
        [
            "diff",
            "--unified=0",
            "--no-ext-diff",
            "--find-renames",
            parent,
            commit
        ]
    )

    if not result["ok"]:

        return {
            "status": (
                "git_timeout"
                if result["timeout"]
                else "git_error"
            ),
            "error": result["stderr"]
        }

    return {
        "status": "success",
        "parent": parent,
        "diff": result["stdout"]
    }


# ============================================================
# BLAME
# ============================================================

def extract_blame_commits(output):
    commits = []

    for line in output.splitlines():

        match = re.match(
            r"^([0-9a-f]{40})\s+\d+\s+\d+",
            line
        )

        if match:
            commits.append(match.group(1))

    return commits


def blame_ranges_once(
    repo,
    parent_commit,
    file_path,
    ranges
):
    """
    One git blame command for one file and multiple ranges.
    """

    if file_path.startswith("/dev/null"):
        return {
            "status": "success",
            "commits": []
        }

    if file_path.startswith("a/"):
        file_path = file_path[2:]

    if file_path.startswith("b/"):
        file_path = file_path[2:]

    args = [
        "blame",
        "--porcelain"
    ]

    for start_line, end_line in ranges:
        args.extend([
            "-L",
            f"{start_line},{end_line}"
        ])

    args.extend([
        parent_commit,
        "--",
        file_path
    ])

    result = run_git_retry(
        repo,
        args
    )

    if not result["ok"]:

        return {
            "status": (
                "git_timeout"
                if result["timeout"]
                else "git_error"
            ),
            "commits": [],
            "error": result["stderr"]
        }

    return {
        "status": "success",
        "commits": extract_blame_commits(
            result["stdout"]
        )
    }


def blame_one_range(
    repo,
    parent_commit,
    file_path,
    start_line,
    end_line
):
    return blame_ranges_once(
        repo,
        parent_commit,
        file_path,
        [(start_line, end_line)]
    )


# ============================================================
# PROCESS ONE FIXING COMMIT
# ============================================================

def process_commit(task):

    project, commit = task

    repo = REPOSITORIES.get(project)

    base = {
        "fix_commit": commit,
        "project": project,
        "inducing_commits": [],
    }

    if repo is None:
        return {
            **base,
            "status": "missing_project",
            "error": "Repository mapping not found"
        }

    commit_data = get_commit_diff(
        repo,
        commit
    )

    if commit_data["status"] != "success":

        return {
            **base,
            "status": commit_data["status"],
            "error": commit_data.get(
                "error",
                ""
            )
        }

    deleted_ranges = parse_deleted_line_ranges(
        commit_data["diff"]
    )

    # This is a VALID SZZ outcome, not a failure.
    if not deleted_ranges:

        return {
            **base,
            "status": "no_deleted_lines",
            "error": "No parent-side deleted/modified lines"
        }

    # --------------------------------------------------------
    # Group ranges by file.
    # --------------------------------------------------------

    grouped = {}

    for file_path, start, end in deleted_ranges:

        grouped.setdefault(
            file_path,
            []
        ).append(
            (start, end)
        )

    inducing = set()
    unresolved_files = []

    # --------------------------------------------------------
    # First attempt: one blame per file.
    # --------------------------------------------------------

    for file_path, ranges in grouped.items():

        result = blame_ranges_once(
            repo,
            commit_data["parent"],
            file_path,
            ranges
        )

        if result["status"] == "success":

            inducing.update(
                result["commits"]
            )

            continue

        # ----------------------------------------------------
        # Fallback: blame each range individually.
        # This handles problematic multi-range blame commands.
        # ----------------------------------------------------

        file_failed = False

        for start_line, end_line in ranges:

            single = blame_one_range(
                repo,
                commit_data["parent"],
                file_path,
                start_line,
                end_line
            )

            if single["status"] == "success":

                inducing.update(
                    single["commits"]
                )

            else:

                file_failed = True

        if file_failed:

            unresolved_files.append({
                "file": file_path,
                "ranges": ranges,
                "error": result.get(
                    "error",
                    "Blame failed"
                )
            })

    # --------------------------------------------------------
    # If ANY range remains unresolved, do NOT use partial
    # results to generate a potentially incorrect buggy label.
    # --------------------------------------------------------

    if unresolved_files:

        return {
            **base,
            "inducing_commits": [],
            "status": "unresolved",
            "error": (
                f"{len(unresolved_files)} file(s) could not "
                f"be completely blamed"
            ),
            "unresolved_files": unresolved_files
        }

    return {
        **base,
        "inducing_commits": sorted(inducing),
        "status": "success"
    }


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(results):

    temp = CHECKPOINT_FILE.with_suffix(
        ".tmp"
    )

    with open(
        temp,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    temp.replace(
        CHECKPOINT_FILE
    )


def load_checkpoint():

    if not CHECKPOINT_FILE.exists():
        return {}

    try:

        with open(
            CHECKPOINT_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        if isinstance(data, dict):
            return data

        return {}

    except Exception as exc:

        print(
            f"WARNING: Could not read checkpoint: {exc}"
        )

        return {}


# ============================================================
# TASK WORKER
# ============================================================

def worker_entry(task, result_queue):
    """
    Each OS process handles exactly ONE task.

    The parent can terminate this process if TASK_TIMEOUT is
    exceeded. This is the key anti-hang mechanism.
    """

    try:

        result = process_commit(task)

        result_queue.put({
            "ok": True,
            "task": task,
            "result": result
        })

    except Exception as exc:

        result_queue.put({
            "ok": False,
            "task": task,
            "result": {
                "fix_commit": task[1],
                "project": task[0],
                "inducing_commits": [],
                "status": "exception",
                "error": repr(exc)
            }
        })


# ============================================================
# FINAL DATASET
# ============================================================

def finalize_dataset(
    df,
    results
):

    print()
    print("=" * 70)
    print("GENERATING FINAL SZZ DATASET")
    print("=" * 70)

    inducing = set()
    status_counts = Counter()

    for result in results.values():

        status = result.get(
            "status",
            "unknown"
        )

        status_counts[status] += 1

        # ONLY fully successful SZZ analyses contribute labels.
        if status == "success":

            inducing.update(
                result.get(
                    "inducing_commits",
                    []
                )
            )

    print(
        f"\nUnique SZZ defect-inducing commits: "
        f"{len(inducing):,}"
    )

    print("\nSZZ result status counts:")

    for status, count in sorted(
        status_counts.items()
    ):

        print(
            f"  {status:<24} {count:,}"
        )

    output = df.copy()

    output["buggy"] = (
        output["commit_id"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {x.lower() for x in inducing}
        )
    )

    output["label_source"] = "szz"
    output["candidate_fix"] = output["fix"]
    output["buggy"] = output["buggy"].astype(bool)

    output.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print(
        f"\nFinal dataset saved to:\n"
        f"{OUTPUT_CSV}"
    )

    print(
        f"\nTotal rows: {len(output):,}"
    )

    print(
        f"Buggy / defect-inducing: "
        f"{output['buggy'].sum():,}"
    )

    print(
        f"Clean: "
        f"{(~output['buggy']).sum():,}"
    )

    print(
        f"Buggy percentage: "
        f"{output['buggy'].mean() * 100:.2f}%"
    )

    print("\nBy language:")

    language_stats = (
        output
        .groupby("language")["buggy"]
        .agg(
            total="count",
            buggy="sum"
        )
    )

    language_stats["buggy_%"] = (
        language_stats["buggy"]
        / language_stats["total"]
        * 100
    )

    print(
        language_stats.to_string()
    )

    print("\nBy project:")

    project_stats = (
        output
        .groupby("project")["buggy"]
        .agg(
            total="count",
            buggy="sum"
        )
    )

    project_stats["buggy_%"] = (
        project_stats["buggy"]
        / project_stats["total"]
        * 100
    )

    print(
        project_stats.to_string()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ROBUST MULTILINGUAL SZZ LABELING v3")
    print("=" * 70)

    start_time = time.time()

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    if not INPUT_CSV.exists():

        print(
            f"\nERROR: Input file not found:\n"
            f"{INPUT_CSV}"
        )

        sys.exit(1)

    print("\nLoading dataset...")

    df = pd.read_csv(
        INPUT_CSV
    )

    print(
        f"Total rows: {len(df):,}"
    )

    required = [
        "commit_id",
        "project",
        "fix"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:

        print(
            "\nERROR: Missing required columns:"
        )

        for c in missing:
            print(
                f"  - {c}"
            )

        sys.exit(1)

    # --------------------------------------------------------
    # Language / repository
    # --------------------------------------------------------

    detected = df["project"].apply(
        detect_language_and_repository
    )

    df["language"] = detected.apply(
        lambda x: x[0]
    )

    df["_repository"] = detected.apply(
        lambda x: x[1]
    )

    print("\nLanguages:")
    print(
        df["language"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nRepositories:")
    print(
        df["_repository"]
        .value_counts(dropna=False)
        .to_string()
    )

    # --------------------------------------------------------
    # Validate repositories once
    # --------------------------------------------------------

    print("\nChecking repositories...")

    invalid_repositories = []

    for name, path in REPOSITORIES.items():

        valid = is_valid_repository(
            path
        )

        print(
            f"{name:<12}"
            f"{'OK' if valid else 'NOT FOUND'}"
        )

        if not valid:
            invalid_repositories.append(
                name
            )

    if invalid_repositories:

        print(
            "\nERROR: Required repositories are missing:"
        )

        for name in invalid_repositories:
            print(
                f"  - {name}"
            )

        sys.exit(1)

    # --------------------------------------------------------
    # Candidate fixes
    # --------------------------------------------------------

    fix_mask = (
        df["fix"]
        .astype(str)
        .str.lower()
        .isin([
            "true",
            "1",
            "yes"
        ])
    )

    fix_df = df[
        fix_mask
    ].copy()

    print(
        f"\nCandidate fixing commits: "
        f"{len(fix_df):,}"
    )

    print("\nCandidate fixes by language:")

    print(
        fix_df["language"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nCandidate fixes by repository:")

    print(
        fix_df["_repository"]
        .value_counts(dropna=False)
        .to_string()
    )

    # --------------------------------------------------------
    # Tasks
    # --------------------------------------------------------

    tasks = []

    for _, row in fix_df.iterrows():

        repo = row["_repository"]

        commit = str(
            row["commit_id"]
        ).strip().lower()

        if repo not in REPOSITORIES:
            continue

        if not re.fullmatch(
            r"[0-9a-f]{40}",
            commit
        ):
            continue

        tasks.append(
            (
                repo,
                commit
            )
        )

    # Remove duplicate task keys while preserving order.
    tasks = list(
        dict.fromkeys(tasks)
    )

    print(
        f"\nValid SZZ tasks: "
        f"{len(tasks):,}"
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    results = load_checkpoint()

    print(
        f"Checkpoint entries found: "
        f"{len(results):,}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Keep old SUCCESS results.
    #
    # Reprocess old statuses that may represent real errors.
    #
    # `no_deleted_lines` is a valid final outcome and does not
    # need to be rerun.
    #
    # `no_parent_or_diff` from the old script is ambiguous,
    # because old Git errors were also collapsed into None.
    # Therefore it IS reprocessed.
    # --------------------------------------------------------

    retry_statuses = {
        "exception",
        "git_error",
        "git_timeout",
        "partial_blame_error",
        "invalid_repository",
        "missing_project",
        "no_parent_or_diff",
    }

    retry_keys = set()

    for key, result in results.items():

        if result.get(
            "status"
        ) in retry_statuses:

            retry_keys.add(key)

    # Remove retryable old entries so they are actually recomputed.
    for key in retry_keys:
        results.pop(
            key,
            None
        )

    if retry_keys:

        print(
            f"Old results marked for reprocessing: "
            f"{len(retry_keys):,}"
        )

    completed = set(
        results.keys()
    )

    remaining = [
        task
        for task in tasks
        if f"{task[0]}::{task[1]}"
        not in completed
    ]

    print(
        f"Remaining tasks: "
        f"{len(remaining):,}"
    )

    # --------------------------------------------------------
    # Complete already
    # --------------------------------------------------------

    if not remaining:

        save_checkpoint(
            results
        )

        with open(
            RESULTS_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                results,
                f,
                indent=2
            )

        finalize_dataset(
            df.drop(
                columns=["_repository"]
            ),
            results
        )

        return

    # --------------------------------------------------------
    # Multiprocessing scheduler
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STARTING TIMEOUT-SAFE SZZ PROCESSING")
    print("=" * 70)

    print(
        f"Workers: {WORKERS}"
    )

    print(
        f"Git timeout: {GIT_TIMEOUT}s"
    )

    print(
        f"Hard task timeout: {TASK_TIMEOUT}s"
    )

    print(
        "Blame strategy: grouped ranges -> individual-range fallback"
    )

    print(
        "Old successful checkpoint results are preserved."
    )

    ctx = mp.get_context("spawn")

    result_queue = ctx.Queue()

    pending = list(remaining)
    active = {}
    processed = 0
    successful = 0
    unresolved = 0
    other_failed = 0

    run_start = time.time()

    try:

        while pending or active:

            # ------------------------------------------------
            # Start workers until capacity is full.
            # ------------------------------------------------

            while pending and len(active) < WORKERS:

                task = pending.pop(0)

                process = ctx.Process(
                    target=worker_entry,
                    args=(
                        task,
                        result_queue
                    )
                )

                process.start()

                active[
                    process.pid
                ] = {
                    "process": process,
                    "task": task,
                    "started": time.time()
                }

            # ------------------------------------------------
            # Collect finished results.
            # ------------------------------------------------

            collected = False

            try:

                message = result_queue.get(
                    timeout=0.5
                )

                collected = True

                task = tuple(
                    message["task"]
                )

                # Find corresponding worker.
                worker_pid = None

                for pid, info in active.items():

                    if info["task"] == task:
                        worker_pid = pid
                        break

                if worker_pid is not None:

                    info = active.pop(
                        worker_pid
                    )

                    process = info["process"]

                    process.join(
                        timeout=1
                    )

                if message["ok"]:

                    result = message["result"]

                else:

                    result = message["result"]

                key = (
                    f"{task[0]}::"
                    f"{task[1]}"
                )

                results[key] = result

                status = result.get(
                    "status",
                    "unknown"
                )

                if status == "success":
                    successful += 1

                elif status == "unresolved":
                    unresolved += 1

                else:
                    other_failed += 1

                processed += 1

                save_checkpoint(
                    results
                )

                elapsed = (
                    time.time()
                    - run_start
                )

                rate = (
                    processed / elapsed
                    if elapsed > 0
                    else 0
                )

                left = (
                    len(remaining)
                    - processed
                )

                eta = (
                    left / rate
                    if rate > 0
                    else 0
                )

                print(
                    f"\rProgress: "
                    f"{processed:,}/"
                    f"{len(remaining):,} "
                    f"({processed / len(remaining) * 100:.1f}%) | "
                    f"Success: {successful:,} | "
                    f"Unresolved: {unresolved:,} | "
                    f"Other failed: {other_failed:,} | "
                    f"Running: {len(active):,} | "
                    f"Rate: {rate:.2f}/s | "
                    f"ETA: {eta / 60:.1f} min",
                    end="",
                    flush=True
                )

            except Exception:
                pass

            # ------------------------------------------------
            # Kill workers exceeding hard task timeout.
            # ------------------------------------------------

            now = time.time()

            timed_out = []

            for pid, info in list(
                active.items()
            ):

                runtime = (
                    now
                    - info["started"]
                )

                if runtime > TASK_TIMEOUT:

                    task = info["task"]
                    process = info["process"]

                    print(
                        f"\nTASK TIMEOUT: "
                        f"{task[0]}::{task[1]} "
                        f"exceeded {TASK_TIMEOUT}s. "
                        f"Terminating worker."
                    )

                    if process.is_alive():
                        process.terminate()

                    process.join(
                        timeout=5
                    )

                    if process.is_alive():

                        # Last resort on Windows.
                        process.kill()

                        process.join(
                            timeout=2
                        )

                    timed_out.append(
                        pid
                    )

                    key = (
                        f"{task[0]}::"
                        f"{task[1]}"
                    )

                    results[key] = {
                        "fix_commit": task[1],
                        "project": task[0],
                        "inducing_commits": [],
                        "status": "unresolved",
                        "error": (
                            f"Entire SZZ task exceeded "
                            f"{TASK_TIMEOUT}s"
                        )
                    }

                    processed += 1
                    unresolved += 1

                    save_checkpoint(
                        results
                    )

                    elapsed = (
                        time.time()
                        - run_start
                    )

                    rate = (
                        processed / elapsed
                        if elapsed > 0
                        else 0
                    )

                    left = (
                        len(remaining)
                        - processed
                    )

                    eta = (
                        left / rate
                        if rate > 0
                        else 0
                    )

                    print(
                        f"Progress: "
                        f"{processed:,}/"
                        f"{len(remaining):,} "
                        f"| Success: {successful:,} "
                        f"| Unresolved: {unresolved:,} "
                        f"| Other failed: {other_failed:,} "
                        f"| Rate: {rate:.2f}/s "
                        f"| ETA: {eta / 60:.1f} min"
                    )

            for pid in timed_out:

                active.pop(
                    pid,
                    None
                )

    except KeyboardInterrupt:

        print(
            "\n\nCtrl+C detected."
        )

        print(
            "Terminating active SZZ workers..."
        )

        for info in active.values():

            process = info["process"]

            if process.is_alive():
                process.terminate()

        for info in active.values():

            process = info["process"]

            process.join(
                timeout=3
            )

            if process.is_alive():
                process.kill()
                process.join(
                    timeout=1
                )

        save_checkpoint(
            results
        )

        print(
            f"Checkpoint saved with "
            f"{len(results):,} entries."
        )

        print(
            "Run v3 again to resume."
        )

        return

    finally:

        for info in active.values():

            process = info["process"]

            if process.is_alive():
                process.terminate()

            process.join(
                timeout=2
            )

    # --------------------------------------------------------
    # Final files
    # --------------------------------------------------------

    save_checkpoint(
        results
    )

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Failure report
    # --------------------------------------------------------

    status_counts = Counter()
    failure_details = {}

    for key, result in results.items():

        status = result.get(
            "status",
            "unknown"
        )

        status_counts[status] += 1

        if status != "success":

            failure_details[key] = result

    with open(
        FAILURE_REPORT,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "status_counts": dict(
                    status_counts
                ),
                "failures": failure_details
            },
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print("SZZ PROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"\nTotal results: "
        f"{len(results):,}"
    )

    print("\nStatus counts:")

    for status, count in sorted(
        status_counts.items()
    ):

        print(
            f"  {status:<24} {count:,}"
        )

    print(
        f"\nFailure/unresolved report saved to:\n"
        f"{FAILURE_REPORT}"
    )

    # --------------------------------------------------------
    # Final dataset
    # --------------------------------------------------------

    finalize_dataset(
        df.drop(
            columns=["_repository"]
        ),
        results
    )

    print(
        f"\nTotal runtime: "
        f"{(time.time() - start_time) / 3600:.2f} hours"
    )

    print("\nDone.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
