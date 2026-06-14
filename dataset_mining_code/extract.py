"""
Commit Extraction Test Script

Purpose:
    Validate commit extraction on a small sample of the ApacheJIT dataset
    before running large-scale extraction.

Workflow:
    1. Load apachejit_total.csv.
    2. Select the first 100 commits as a test sample.
    3. Traverse the corresponding Git repositories using PyDriller.
    4. Match commits listed in the sample dataset.
    5. Extract:
         - Commit hash
         - Commit message
         - Source code diff (patch)
    6. Store extracted information in a pandas DataFrame.
    7. Print sample results and the total number of extracted commits.

Input:
    - apachejit_total.csv
    - Local Git repositories under repo_base

Output:
    - In-memory DataFrame (df_git) containing:
        * commit_id
        * message
        * diff
    - No files are written to disk.

Notes:
    - Intended for testing and verification only.
    - Uses only the first 100 dataset entries to reduce execution time.
    - Helps confirm repository paths, commit matching, and PyDriller
      extraction before processing the full dataset.
"""
import pandas as pd
from pydriller import Repository

# Load dataset
df = pd.read_csv("apachejit_total.csv")

# Take small sample (100 commits)
df_sample = df.head(100)

repo_base = "D:/apachejit/repos/"

def get_repo_path(project_name):
    return repo_base + project_name.split("/")[1]

data = []

for project in df_sample["project"].unique():
    repo_path = get_repo_path(project)

    print(f"\nProcessing {project}...")

    project_commits = set(
        df_sample[df_sample["project"] == project]["commit_id"]
    )

    for commit in Repository(repo_path).traverse_commits():
        if commit.hash in project_commits:
            print(f"Found: {commit.hash}")

            data.append({
                "commit_id": commit.hash,
                "message": commit.msg,
                "diff": "\n".join([m.diff for m in commit.modified_files])
            })

df_git = pd.DataFrame(data)

print("\nExtracted:")
print(df_git.head())
print("Total extracted:", len(df_git))