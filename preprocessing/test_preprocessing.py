import logging

from preprocessing.load_data import load_commits_csv
from preprocessing.clean_data import clean_commits_df
from preprocessing.sampling import create_stratified_subset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

df = load_commits_csv("final_multilanguage_dataset.csv")

jit_columns = [
    "la", "ld", "nf", "ns", "nd", "ent",
    "ndev", "age", "nuc", "aexp", "arexp", "asexp",
]

df_clean, report = clean_commits_df(df, jit_feature_columns=jit_columns)
sampled = create_stratified_subset(df_clean)

print("cleaned rows:", len(df_clean))
print("sampled rows:", len(sampled))
print("cleaning report:", report)
print("sample counts by project × buggy:")
print(sampled.groupby(["project", "buggy"]).size().sort_index())