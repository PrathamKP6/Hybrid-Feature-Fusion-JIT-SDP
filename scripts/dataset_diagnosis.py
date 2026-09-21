import pandas as pd

MASTER = "data/final_multilanguage_59996_szz_buggy.csv"
LLM_FILES = [
    "data/llm_experiment_dataset/train.csv",
    "data/llm_experiment_dataset/validation.csv",
    "data/llm_experiment_dataset/test.csv",
]

# ------------------------------------------------------------
# Load canonical labels
# ------------------------------------------------------------

master = pd.read_csv(
    MASTER,
    usecols=["commit_id", "buggy"],
    low_memory=False,
)

master["commit_id"] = master["commit_id"].astype(str)

# Normalize canonical labels to strings for comparison
master["buggy"] = master["buggy"].astype(str).str.strip()

# ------------------------------------------------------------
# Load LLM files
# ------------------------------------------------------------

llm_list = []

for path in LLM_FILES:
    df = pd.read_csv(
        path,
        usecols=["commit_id", "buggy"],
        low_memory=False,
    )

    df["commit_id"] = df["commit_id"].astype(str)
    df["buggy"] = df["buggy"].astype(str).str.strip()

    df["source"] = path

    llm_list.append(df)

llm = pd.concat(
    llm_list,
    ignore_index=True,
)

# ------------------------------------------------------------
# Compare
# ------------------------------------------------------------

comparison = master.merge(
    llm,
    on="commit_id",
    how="inner",
    suffixes=("_canonical", "_llm"),
)

comparison["match"] = (
    comparison["buggy_canonical"]
    == comparison["buggy_llm"]
)

print("=" * 70)
print("LABEL DIAGNOSTIC")
print("=" * 70)

print(f"Canonical rows : {len(master):,}")
print(f"LLM rows       : {len(llm):,}")
print(f"Matched IDs    : {len(comparison):,}")

print(
    f"\nMatching labels    : "
    f"{comparison['match'].sum():,}"
)

print(
    f"Mismatching labels : "
    f"{(~comparison['match']).sum():,}"
)

print("\n" + "=" * 70)
print("CANONICAL LABEL DISTRIBUTION")
print("=" * 70)
print(master["buggy"].value_counts(dropna=False))

print("\n" + "=" * 70)
print("LLM LABEL DISTRIBUTION")
print("=" * 70)
print(llm["buggy"].value_counts(dropna=False))

print("\n" + "=" * 70)
print("LABEL CROSS-TABULATION")
print("=" * 70)

print(
    pd.crosstab(
        comparison["buggy_canonical"],
        comparison["buggy_llm"],
        margins=True,
    )
)

print("\n" + "=" * 70)
print("MISMATCH EXAMPLES")
print("=" * 70)

mismatches = comparison[
    ~comparison["match"]
]

print(
    mismatches.head(30).to_string(index=False)
)