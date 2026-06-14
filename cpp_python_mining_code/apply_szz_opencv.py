"""
Simplified SZZ Algorithm Implementation for OpenCV Dataset
Relabels buggy commits by tracing fix commits back to their root causes using Git blame.
Optimized with subprocess calls for faster blame operations.
"""

import pandas as pd
from pathlib import Path
from git import Repo
from git.exc import GitCommandError
import subprocess
import json

# Configuration
REPO_PATH = "opencv"
INPUT_CSV = "cpp_opencv_dataset.csv"
OUTPUT_CSV = "cpp_opencv_dataset_szz.csv"
PROGRESS_INTERVAL = 50
FIX_KEYWORDS = ["fix", "bug", "error", "issue", "patch", "resolve", "hotfix"]


def is_fix_commit(message):
    """
    Check if a commit is a fix commit based on keywords.
    
    Args:
        message: Commit message string
    
    Returns:
        True if message contains any fix keyword, False otherwise
    """
    if not isinstance(message, str):
        return False
    text = message.lower()
    return any(keyword in text for keyword in FIX_KEYWORDS)


def load_dataset(csv_path):
    """
    Load the dataset from CSV file.
    
    Args:
        csv_path: Path to the CSV file
    
    Returns:
        DataFrame with dataset
    """
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} commits")
    return df


def label_fix_commits(df):
    """
    Label fix commits based on message keywords.
    
    Args:
        df: DataFrame with commit data
    
    Returns:
        DataFrame with updated 'fix' column
    """
    df["fix"] = df["message"].apply(lambda x: 1 if is_fix_commit(x) else 0)
    num_fix = df["fix"].sum()
    print(f"Identified {num_fix} fix commits")
    return df


def load_repository(repo_path):
    """
    Load the Git repository from local path.
    
    Args:
        repo_path: Path to the Git repository
    
    Returns:
        GitPython Repo object
    """
    print(f"Loading repository from {repo_path}...")
    repo = Repo(repo_path)
    try:
        origin_url = repo.remotes.origin.url if repo.remotes else "local repo"
        print(f"Repository loaded: {origin_url}")
    except:
        print(f"Repository loaded from {repo_path}")
    return repo


def apply_szz_algorithm(df, repo):
    """
    Apply simplified SZZ algorithm using subprocess git blame for better performance.
    
    Args:
        df: DataFrame with commit data
        repo: GitPython Repo object
    
    Returns:
        Set of buggy commit hashes
    """
    buggy_commits = set()
    fix_commits = df[df["fix"] == 1]["commit_id"].tolist()
    cpp_extensions = {".cpp", ".cc", ".hpp", ".h", ".cxx", ".c"}
    
    print(f"\nProcessing {len(fix_commits)} fix commits for SZZ algorithm...")
    print(f"(Using subprocess git blame for performance)\n")
    
    for idx, commit_id in enumerate(fix_commits):
        if (idx + 1) % PROGRESS_INTERVAL == 0 or idx == 0:
            print(f"  [{idx + 1:4d}/{len(fix_commits)}] Found {len(buggy_commits):5d} buggy commits")
        
        try:
            commit = repo.commit(commit_id)
        except Exception:
            continue
        
        # Skip commits with no parents
        if not commit.parents:
            continue
        
        # Process only first parent (simplified)
        parent = commit.parents[0]
        
        try:
            # Get diff between parent and current commit
            diffs = parent.diff(commit, create_patch=True)
            
            # Limit diffs processed per commit (performance optimization)
            diff_count = 0
            for diff in diffs:
                if diff_count > 20:  # Skip if too many files changed
                    break
                    
                file_path = diff.b_path
                if file_path is None:
                    continue
                
                # Only blame C++ source files
                file_ext = Path(file_path).suffix.lower()
                if file_ext not in cpp_extensions:
                    continue
                
                diff_count += 1
                
                try:
                    # Use subprocess git blame for better performance
                    blamed_commits = get_blame_commits(
                        repo.working_dir, 
                        parent.hexsha, 
                        file_path
                    )
                    buggy_commits.update(blamed_commits)
                
                except Exception:
                    # Skip files that can't be blamed
                    pass
        
        except Exception:
            continue
    
    print(f"\n✓ SZZ algorithm complete: Found {len(buggy_commits)} unique buggy commits")
    return buggy_commits


def get_blame_commits(repo_path, commit_sha, file_path):
    """
    Get blamed commit SHAs using subprocess git blame with timeout.
    
    Args:
        repo_path: Path to repository
        commit_sha: SHA of commit to blame against
        file_path: Path to file
    
    Returns:
        Set of blamed commit SHAs
    """
    blamed = set()
    try:
        # Use git blame with porcelain format for easy parsing
        cmd = ["git", "blame", "--porcelain", f"{commit_sha}~1", "--", file_path]
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            encoding='utf-8',
            errors='ignore',
            timeout=10  # 10 second timeout per file
        )
        
        if result.returncode != 0:
            return blamed
        
        # Parse porcelain format: first line has commit SHA
        for line in result.stdout.split('\n'):
            if line and not line.startswith('\t'):
                # Extract commit SHA (first 40 chars)
                parts = line.split()
                if parts:
                    sha = parts[0]
                    if len(sha) == 40:  # Valid SHA
                        blamed.add(sha)
    
    except (subprocess.TimeoutExpired, Exception):
        pass
    
    return blamed


def label_buggy_commits(df, buggy_commits):
    """
    Label commits as buggy based on SZZ results.
    
    Args:
        df: DataFrame with commit data
        buggy_commits: Set of buggy commit hashes
    
    Returns:
        DataFrame with updated 'buggy' column
    """
    df["buggy"] = df["commit_id"].apply(
        lambda x: 1 if x in buggy_commits else 0
    )
    num_buggy = df["buggy"].sum()
    print(f"Labeled {num_buggy} commits as buggy")
    return df


def save_dataset(df, output_path):
    """
    Save the modified dataset to CSV file.
    
    Args:
        df: DataFrame to save
        output_path: Path to output CSV file
    """
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\nBuggy distribution:\n{df['buggy'].value_counts().to_dict()}")
    print(f"\nFix distribution:\n{df['fix'].value_counts().to_dict()}")


def main():
    """Main execution flow for SZZ algorithm."""
    print("=" * 80)
    print("Simplified SZZ Algorithm - OpenCV Dataset")
    print("=" * 80 + "\n")
    
    # Step 1: Load dataset
    df = load_dataset(INPUT_CSV)
    
    # Step 2: Label fix commits
    df = label_fix_commits(df)
    
    # Step 3: Load repository
    repo = load_repository(REPO_PATH)
    
    # Step 4: Apply SZZ algorithm
    buggy_commits = apply_szz_algorithm(df, repo)
    
    # Step 5: Label buggy commits
    df = label_buggy_commits(df, buggy_commits)
    
    # Step 6: Save output
    save_dataset(df, OUTPUT_CSV)
    
    # Debug summary
    print("\n" + "=" * 80)
    print("Execution Summary")
    print("=" * 80)
    print(f"Total commits:     {len(df)}")
    print(f"Fix commits:       {df['fix'].sum()}")
    print(f"Buggy commits:     {df['buggy'].sum()}")
    print(f"Non-buggy commits: {(df['buggy'] == 0).sum()}")
    print(f"\nOutput file: {OUTPUT_CSV}")
    print("=" * 80)


if __name__ == "__main__":
    main()
