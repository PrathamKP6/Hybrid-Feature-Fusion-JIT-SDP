"""
Dataset Merge Script

Purpose:
    Combine ApacheJIT commit metadata and bug labels with extracted
    commit messages and code diffs using commit_id as the key.

Input:
    - apachejit_total.csv
    - extracted_commits.csv

Output:
    - final_dataset.csv

Result:
    Produces a unified dataset containing commit labels, project
    information, commit messages, and code diffs for downstream
    defect prediction experiments.
"""
import pandas as pd

print("Loading ApacheJIT dataset...")
df_main = pd.read_csv("apachejit_total.csv")

print("Loading extracted commits...")
df_extra = pd.read_csv("extracted_commits.csv")

print("Merging datasets...")

df_final = df_main.merge(df_extra, on="commit_id", how="inner")

print("Final dataset shape:", df_final.shape)

# Check missing values
print("\nMissing values:")
print(df_final.isnull().sum())

# Save final dataset
df_final.to_csv("final_dataset.csv", index=False)

print("\nSaved as final_dataset.csv")