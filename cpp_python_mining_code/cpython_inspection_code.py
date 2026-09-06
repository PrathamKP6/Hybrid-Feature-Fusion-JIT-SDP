import os
from collections import Counter

from git import Repo


# ============================================================
# CONFIG
# ============================================================

CPYTHON_PATH = r"D:\embeddings_and_pca_code\cpp_python_mining_code\cpython"

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


# ============================================================
# HELPERS
# ============================================================

def is_python_file(path):
    if not path:
        return False

    _, ext = os.path.splitext(path.lower())

    return ext in PYTHON_EXTENSIONS


def is_bug_fix_commit(message):
    if not message:
        return False

    message_lower = message.lower()

    return any(
        keyword in message_lower
        for keyword in BUG_KEYWORDS
    )


# ============================================================
# INSPECT CPYTHON
# ============================================================

def inspect_cpython():

    print("=" * 72)
    print("CPYTHON PYTHON-COMMIT INSPECTION")
    print("=" * 72)

    if not os.path.isdir(CPYTHON_PATH):
        raise FileNotFoundError(
            f"CPython repository not found:\n{CPYTHON_PATH}"
        )

    repo = Repo(CPYTHON_PATH)

    print(f"\nRepository:")
    print(CPYTHON_PATH)

    print("\nScanning Git history...")
    print("A commit is counted if it touched AT LEAST ONE .py file.")
    print("The commit may also touch C/C++/header/other files.\n")

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total_commits = 0

    python_commits = 0

    python_bugfix_commits = 0

    python_non_bugfix_commits = 0

    python_only_commits = 0

    python_and_other_commits = 0

    total_python_files_touched = 0

    python_file_extensions = Counter()

    # --------------------------------------------------------
    # Keep some examples for verification
    # --------------------------------------------------------

    examples = []

    # --------------------------------------------------------
    # Iterate through complete Git history
    # --------------------------------------------------------

    for commit in repo.iter_commits("--all"):

        total_commits += 1

        python_files = set()
        other_files = set()

        try:

            # ------------------------------------------------
            # Normal commit with parent
            # ------------------------------------------------

            if commit.parents:

                parent = commit.parents[0]

                diffs = parent.diff(commit)

                for diff in diffs:

                    old_path = diff.a_path
                    new_path = diff.b_path

                    paths = {
                        p
                        for p in (
                            old_path,
                            new_path,
                        )
                        if p
                    }

                    for path in paths:

                        if is_python_file(path):

                            python_files.add(path)

                        else:

                            other_files.add(path)

            # ------------------------------------------------
            # Initial commit
            # ------------------------------------------------

            else:

                for path in commit.stats.files:

                    if is_python_file(path):

                        python_files.add(path)

                    else:

                        other_files.add(path)

        except Exception as e:

            print(
                f"\n[WARNING] Could not inspect "
                f"commit {commit.hexsha}: {e}"
            )

            continue

        # ----------------------------------------------------
        # Commit is relevant if it touched >= 1 Python file.
        # ----------------------------------------------------

        if python_files:

            python_commits += 1

            total_python_files_touched += len(
                python_files
            )

            # ------------------------------------------------
            # Bug/fix keyword classification
            # ------------------------------------------------

            if is_bug_fix_commit(commit.message):

                python_bugfix_commits += 1

            else:

                python_non_bugfix_commits += 1

            # ------------------------------------------------
            # Python-only vs Python + other files
            # ------------------------------------------------

            if other_files:

                python_and_other_commits += 1

            else:

                python_only_commits += 1

            # ------------------------------------------------
            # Extension counts
            # ------------------------------------------------

            for path in python_files:

                _, ext = os.path.splitext(
                    path.lower()
                )

                python_file_extensions[ext] += 1

            # ------------------------------------------------
            # Save first 20 examples
            # ------------------------------------------------

            if len(examples) < 20:

                examples.append(
                    {
                        "hash": commit.hexsha,
                        "date": commit.committed_datetime,
                        "message": (
                            commit.message
                            .splitlines()[0]
                            if commit.message
                            else ""
                        ),
                        "python_files": len(
                            python_files
                        ),
                        "other_files": len(
                            other_files
                        ),
                        "bugfix": is_bug_fix_commit(
                            commit.message
                        ),
                    }
                )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if total_commits % 1000 == 0:

            print(
                f"Processed {total_commits} commits | "
                f"Python commits found: {python_commits}"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n")
    print("=" * 72)
    print("CPYTHON INSPECTION COMPLETE")
    print("=" * 72)

    print(
        f"\nTotal Git commits inspected : "
        f"{total_commits:,}"
    )

    print(
        f"Commits touching .py files : "
        f"{python_commits:,}"
    )

    print(
        f"Python + other files        : "
        f"{python_and_other_commits:,}"
    )

    print(
        f"Python-only commits         : "
        f"{python_only_commits:,}"
    )

    print(
        f"\nBug/fix keyword commits     : "
        f"{python_bugfix_commits:,}"
    )

    print(
        f"Non-bug/fix commits         : "
        f"{python_non_bugfix_commits:,}"
    )

    print(
        f"\nTotal .py file touches      : "
        f"{total_python_files_touched:,}"
    )

    # --------------------------------------------------------
    # Can we get 12,000 rows?
    # --------------------------------------------------------

    print("\n" + "-" * 72)

    if python_commits >= 12000:

        print(
            f"YES: At least 12,000 commits touch Python files."
        )

        print(
            f"Available: {python_commits:,}"
        )

        print(
            f"Required:  12,000"
        )

        print(
            f"Surplus:   {python_commits - 12000:,}"
        )

    else:

        print(
            f"NO: Fewer than 12,000 commits touch Python files."
        )

        print(
            f"Available: {python_commits:,}"
        )

        print(
            f"Required:  12,000"
        )

        print(
            f"Shortfall: {12000 - python_commits:,}"
        )

    # --------------------------------------------------------
    # Bug/fix availability
    # --------------------------------------------------------

    print("\n" + "-" * 72)

    print("BUG/FIX KEYWORD POOL")

    print(
        f"Commits touching .py + bug/fix keyword: "
        f"{python_bugfix_commits:,}"
    )

    print(
        f"Commits touching .py without keyword: "
        f"{python_non_bugfix_commits:,}"
    )

    # --------------------------------------------------------
    # File extension statistics
    # --------------------------------------------------------

    print("\n" + "-" * 72)

    print("PYTHON FILE EXTENSIONS")

    for ext, count in sorted(
        python_file_extensions.items()
    ):

        print(
            f"{ext}: {count:,} file touches"
        )

    # --------------------------------------------------------
    # Example commits
    # --------------------------------------------------------

    print("\n" + "-" * 72)

    print("FIRST 20 PYTHON-TOUCHING COMMITS")

    for i, item in enumerate(
        examples,
        start=1,
    ):

        print(
            f"\n{i}. {item['hash']}"
        )

        print(
            f"   Date         : {item['date']}"
        )

        print(
            f"   Python files : {item['python_files']}"
        )

        print(
            f"   Other files  : {item['other_files']}"
        )

        print(
            f"   Bug/fix      : {item['bugfix']}"
        )

        print(
            f"   Message      : {item['message']}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    inspect_cpython()