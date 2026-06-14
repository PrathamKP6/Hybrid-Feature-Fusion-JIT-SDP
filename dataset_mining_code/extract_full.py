"""
Commit Extraction Script

Purpose:
    Extract commit messages and source code diffs for commits listed in
    apachejit_total.csv using PyDriller.

Input:
    - apachejit_total.csv containing project names and commit IDs
    - Local cloned repositories stored under repo_base

Output:
    - extracted_commits.csv containing:
        * commit_id
        * commit message
        * combined code diff

Notes:
    - Supports resuming interrupted executions by skipping already
      extracted commits present in extracted_commits.csv.
    - Intended for dataset construction and preprocessing.
    - Typically run once during dataset generation; not required during
      model training or evaluation.
"""
import pandas as pd
from pydriller import Repository
import csv
import os

# Load dataset
df = pd.read_csv("apachejit_total.csv")

repo_base = "D:/apachejit/repos/"

def get_repo_path(project_name):
    return repo_base + project_name.split("/")[1]

output_file = "extracted_commits.csv"

# Load already extracted (resume support)
if os.path.exists(output_file):
    df_existing = pd.read_csv(output_file)
    done_commits = set(df_existing["commit_id"])
    print(f"Resuming... already have {len(done_commits)} commits")
else:
    done_commits = set()

data = []

for project in df["project"].unique():
    repo_path = get_repo_path(project)

    print(f"\nProcessing {project}...")

    project_df = df[df["project"] == project]
    project_commits = set(project_df["commit_id"]) - done_commits

    for commit in Repository(repo_path).traverse_commits():
        if commit.hash in project_commits:

            data.append({
                "commit_id": commit.hash,
                "message": commit.msg,
                "diff": "\n".join([m.diff for m in commit.modified_files])
            })

            # ✅ Save every 500 commits (FIXED INDENTATION)
            if len(data) >= 500:
                temp_df = pd.DataFrame(data)

                temp_df.to_csv(
                    output_file,
                    mode="a",
                    index=False,
                    header=not os.path.exists(output_file),
                    quoting=csv.QUOTE_ALL,
                    escapechar="\\"
                )

                print(f"Saved {len(data)} commits...")
                data = []

# ✅ Final save
if data:
    temp_df = pd.DataFrame(data)

    temp_df.to_csv(
        output_file,
        mode="a",
        index=False,
        header=not os.path.exists(output_file),
        quoting=csv.QUOTE_ALL,
        escapechar="\\"
    )

    print(f"Saved final {len(data)} commits...")

print("Extraction completed!")