from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import pandas as pd
import json
import re
import time
import sys
from collections import Counter

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

LANGUAGE_PREFIXES = {"python": "python", "cpp": "cpp", "apache": "java"}

WORKERS = 4
CHECKPOINT_EVERY = 25
GIT_TIMEOUT = 45
BATCH_SIZE = WORKERS * 4


def run_git(repo, args, timeout=GIT_TIMEOUT):
    command = ["git", "-C", str(repo)] + args
    try:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr.strip(), False
    except subprocess.TimeoutExpired:
        return False, "", "Git command timed out", True
    except Exception as exc:
        return False, "", str(exc), False


def is_valid_repository(repo):
    ok, stdout, _, _ = run_git(repo, ["rev-parse", "--is-inside-work-tree"])
    return ok and stdout.strip() == "true"


def detect_language_and_repository(project):
    if pd.isna(project):
        return None, None
    parts = [p.strip() for p in str(project).strip().replace("\\", "/").split("/") if p.strip()]
    if not parts:
        return None, None
    language = LANGUAGE_PREFIXES.get(parts[0].lower())
    repository = parts[1].lower() if len(parts) >= 2 else None
    return language, repository


def parse_deleted_line_ranges(diff_text):
    ranges = []
    if not diff_text:
        return ranges
    current_file = None
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            match = re.match(r"diff --git a/(.*?) b/(.*)", line)
            current_file = match.group(1) if match else None
            continue
        if line.startswith("Binary files"):
            current_file = None
            continue
        if line.startswith("@@"):
            match = re.match(r"@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", line)
            if not match or current_file is None:
                continue
            old_start = int(match.group(1))
            old_count = int(match.group(2) or 1)
            if old_count > 0:
                ranges.append((current_file, old_start, old_start + old_count - 1))
    return ranges


def get_commit_diff(repo, commit):
    ok, parents, stderr, timed_out = run_git(
        repo, ["rev-list", "--parents", "-n", "1", commit]
    )
    if not ok:
        return {"status": "git_timeout" if timed_out else "git_error",
                "error": stderr or "git rev-list failed"}
    parts = parents.strip().split()
    if len(parts) < 2:
        return {"status": "no_parent_or_diff", "error": "Commit has no parent"}
    parent = parts[1]
    ok, diff, stderr, timed_out = run_git(
        repo, ["diff", "--unified=0", "--no-ext-diff", "--find-renames",
               parent, commit]
    )
    if not ok:
        return {"status": "git_timeout" if timed_out else "git_error",
                "error": stderr or "git diff failed"}
    return {"status": "success", "parent": parent, "diff": diff}


def blame_range(repo, parent_commit, file_path, start_line, end_line):
    if file_path.startswith("/dev/null"):
        return {"status": "success", "commits": []}
    if file_path.startswith("b/"):
        file_path = file_path[2:]
    if file_path.startswith("a/"):
        file_path = file_path[2:]

    ok, output, stderr, timed_out = run_git(
        repo,
        ["blame", "--porcelain", "-L", f"{start_line},{end_line}",
         parent_commit, "--", file_path]
    )
    if not ok:
        return {"status": "git_timeout" if timed_out else "git_error",
                "commits": [], "error": stderr or "git blame failed"}

    commits = []
    for line in output.splitlines():
        match = re.match(r"^([0-9a-f]{40})\s+\d+\s+\d+", line)
        if match:
            commits.append(match.group(1))
    return {"status": "success", "commits": commits}


def process_commit(task):
    project, commit = task
    repo = REPOSITORIES.get(project)
    base = {"fix_commit": commit, "project": project, "inducing_commits": []}

    if repo is None:
        return {**base, "status": "missing_project"}

    commit_data = get_commit_diff(repo, commit)
    if commit_data["status"] != "success":
        return {**base, "status": commit_data["status"],
                "error": commit_data.get("error", "")}

    deleted_ranges = parse_deleted_line_ranges(commit_data["diff"])
    if not deleted_ranges:
        return {**base, "status": "no_deleted_lines"}

    inducing = set()
    blame_errors = []

    for file_path, start_line, end_line in deleted_ranges:
        blame = blame_range(repo, commit_data["parent"], file_path, start_line, end_line)
        if blame["status"] != "success":
            blame_errors.append({
                "file": file_path, "start": start_line, "end": end_line,
                "status": blame["status"], "error": blame.get("error", "")
            })
            continue
        for sha in blame["commits"]:
            if sha != "0" * 40 and sha != commit:
                inducing.add(sha)

    if blame_errors:
        return {
            **base, "inducing_commits": sorted(inducing),
            "status": "partial_blame_error",
            "error": f"{len(blame_errors)} blame operation(s) failed",
            "blame_errors": blame_errors
        }

    return {**base, "inducing_commits": sorted(inducing), "status": "success"}


def save_checkpoint(results):
    temp = CHECKPOINT_FILE.with_suffix(".tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    temp.replace(CHECKPOINT_FILE)


def load_checkpoint():
    if not CHECKPOINT_FILE.exists():
        return {}
    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"WARNING: Could not read checkpoint: {exc}")
        return {}


def finalize_dataset(df, results):
    print("\n" + "=" * 70)
    print("GENERATING FINAL SZZ DATASET")
    print("=" * 70)

    inducing = set()
    status_counts = Counter()

    for result in results.values():
        status = result.get("status", "unknown")
        status_counts[status] += 1
        if status in ("success", "partial_blame_error"):
            inducing.update(result.get("inducing_commits", []))

    print(f"\nUnique SZZ defect-inducing commits: {len(inducing):,}")
    print("\nSZZ result status counts:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status:<24} {count:,}")

    output = df.copy()
    output["buggy"] = output["commit_id"].astype(str).str.strip().isin(inducing)
    output["label_source"] = "szz"
    output["candidate_fix"] = output["fix"]
    output["buggy"] = output["buggy"].astype(bool)
    output.to_csv(OUTPUT_CSV, index=False)

    print(f"\nFinal dataset saved to:\n{OUTPUT_CSV}")
    print(f"Total rows: {len(output):,}")
    print(f"Buggy / defect-inducing: {output['buggy'].sum():,}")
    print(f"Clean: {(~output['buggy']).sum():,}")
    print(f"Buggy percentage: {output['buggy'].mean() * 100:.2f}%")

    print("\nBy language:")
    language_stats = output.groupby("language")["buggy"].agg(total="count", buggy="sum")
    language_stats["buggy_%"] = language_stats["buggy"] / language_stats["total"] * 100
    print(language_stats.to_string())

    print("\nBy project:")
    project_stats = output.groupby("project")["buggy"].agg(total="count", buggy="sum")
    project_stats["buggy_%"] = project_stats["buggy"] / project_stats["total"] * 100
    print(project_stats.to_string())


def main():
    print("=" * 70)
    print("ROBUST MULTILINGUAL SZZ LABELING")
    print("=" * 70)

    start_time = time.time()

    if not INPUT_CSV.exists():
        print(f"ERROR: Input file not found:\n{INPUT_CSV}")
        sys.exit(1)

    print("\nLoading dataset...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Total rows: {len(df):,}")

    required = ["commit_id", "project", "fix"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("ERROR: Missing required columns:")
        for c in missing:
            print(f"  - {c}")
        sys.exit(1)

    detected = df["project"].apply(detect_language_and_repository)
    df["language"] = detected.apply(lambda x: x[0])
    df["_repository"] = detected.apply(lambda x: x[1])

    print("\nLanguages:")
    print(df["language"].value_counts(dropna=False).to_string())

    print("\nRepositories:")
    print(df["_repository"].value_counts(dropna=False).to_string())

    print("\nChecking repositories...")
    for name, path in REPOSITORIES.items():
        valid = is_valid_repository(path)
        print(f"{name:<12}{'OK' if valid else 'NOT FOUND'}")

    fix_mask = df["fix"].astype(str).str.lower().isin(["true", "1", "yes"])
    fix_df = df[fix_mask].copy()

    print(f"\nCandidate fixing commits: {len(fix_df):,}")
    print("\nCandidate fixes by language:")
    print(fix_df["language"].value_counts(dropna=False).to_string())
    print("\nCandidate fixes by repository:")
    print(fix_df["_repository"].value_counts(dropna=False).to_string())

    tasks = []
    for _, row in fix_df.iterrows():
        repo = row["_repository"]
        commit = str(row["commit_id"]).strip().lower()
        if repo in REPOSITORIES and re.fullmatch(r"[0-9a-f]{40}", commit):
            tasks.append((repo, commit))

    print(f"\nValid SZZ tasks: {len(tasks):,}")

    results = load_checkpoint()
    print(f"Checkpoint entries found: {len(results):,}")

    completed = set(results.keys())
    remaining = [t for t in tasks if f"{t[0]}::{t[1]}" not in completed]

    print(f"Remaining tasks: {len(remaining):,}")

    if not remaining:
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        finalize_dataset(df.drop(columns=["_repository"]), results)
        return

    print("\n" + "=" * 70)
    print("RESUMING SZZ PROCESSING")
    print("=" * 70)
    print(f"Workers: {WORKERS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Git timeout: {GIT_TIMEOUT}s")

    processed = successful = failed = 0
    run_start = time.time()

    try:
        for batch_start in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[batch_start:batch_start + BATCH_SIZE]

            print(f"\nStarting batch {batch_start + 1:,}-"
                  f"{batch_start + len(batch):,} of {len(remaining):,}")

            with ThreadPoolExecutor(max_workers=WORKERS) as executor:
                future_map = {executor.submit(process_commit, t): t for t in batch}

                for future in as_completed(future_map):
                    task = future_map[future]
                    key = f"{task[0]}::{task[1]}"

                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            "fix_commit": task[1], "project": task[0],
                            "inducing_commits": [], "status": "exception",
                            "error": str(exc)
                        }

                    results[key] = result

                    if result.get("status") == "success":
                        successful += 1
                    else:
                        failed += 1

                    processed += 1
                    elapsed = time.time() - run_start
                    rate = processed / elapsed if elapsed > 0 else 0
                    left = len(remaining) - processed
                    eta = left / rate if rate > 0 else 0

                    print(
                        f"\rProgress: {processed:,}/{len(remaining):,} "
                        f"({processed / len(remaining) * 100:.1f}%) | "
                        f"Success: {successful:,} | Failed: {failed:,} | "
                        f"Rate: {rate:.2f}/s | ETA: {eta / 60:.1f} min",
                        end="", flush=True
                    )

                    if processed % CHECKPOINT_EVERY == 0:
                        save_checkpoint(results)
                        print("\nCheckpoint saved.")

            save_checkpoint(results)
            print("\nBatch checkpoint saved.")

    except KeyboardInterrupt:
        print("\n\nCtrl+C detected. Saving completed results...")
        save_checkpoint(results)
        print(f"Checkpoint saved with {len(results):,} entries.")
        print("Run the same script again to resume.")
        return

    save_checkpoint(results)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    status_counts = Counter()
    failure_details = {}

    for key, result in results.items():
        status = result.get("status", "unknown")
        status_counts[status] += 1
        if status != "success":
            failure_details[key] = result

    with open(FAILURE_REPORT, "w", encoding="utf-8") as f:
        json.dump({"status_counts": dict(status_counts),
                   "failures": failure_details}, f, indent=2)

    print("\n" + "=" * 70)
    print("SZZ PROCESSING COMPLETE")
    print("=" * 70)

    for status, count in sorted(status_counts.items()):
        print(f"  {status:<24} {count:,}")

    print(f"\nFailure report saved to:\n{FAILURE_REPORT}")

    finalize_dataset(df.drop(columns=["_repository"]), results)

    print(f"\nTotal runtime this run: {(time.time() - start_time) / 3600:.2f} hours")
    print("Done.")


if __name__ == "__main__":
    main()
