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