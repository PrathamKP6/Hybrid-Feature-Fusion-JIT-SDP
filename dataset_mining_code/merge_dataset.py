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