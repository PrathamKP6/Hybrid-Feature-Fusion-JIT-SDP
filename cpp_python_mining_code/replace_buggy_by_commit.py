import pandas as pd
import shutil
import os

# ============================================================
# FILE PATHS
# ============================================================

SZZ_FILE = r"D:\embeddings_and_pca_code\cpp_python_mining_code\python_cpp_23996_szz.csv"

MAIN_FILE = r"D:\embeddings_and_pca_code\data\final_multilanguage_59996_768d_reconstructed.csv"

OUTPUT_FILE = r"D:\embeddings_and_pca_code\data\final_multilanguage_59996_szz_buggy.csv"

BACKUP_FILE = r"D:\embeddings_and_pca_code\data\final_multilanguage_59996_768d_reconstructed_backup.csv"


# ============================================================
# START
# ============================================================

print("=" * 75)
print("SZZ BUGGY LABEL REPLACEMENT")
print("=" * 75)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(SZZ_FILE):
    raise FileNotFoundError(f"SZZ file not found:\n{SZZ_FILE}")

if not os.path.exists(MAIN_FILE):
    raise FileNotFoundError(f"Main dataset not found:\n{MAIN_FILE}")


# ============================================================
# CREATE BACKUP
# ============================================================

print("\nCreating backup of original 59996 dataset...")

if not os.path.exists(BACKUP_FILE):
    shutil.copy2(MAIN_FILE, BACKUP_FILE)
    print(f"Backup created:")
    print(BACKUP_FILE)
else:
    print("Backup already exists. Not overwriting it.")


# ============================================================
# READ SZZ DATASET
# ============================================================

print("\nReading SZZ dataset...")

szz = pd.read_csv(
    SZZ_FILE,
    usecols=["commit_id", "buggy"]
)

print(f"SZZ rows: {len(szz):,}")


# ============================================================
# READ MAIN DATASET
# ============================================================

print("\nReading 59996 dataset...")

df = pd.read_csv(MAIN_FILE)

print(f"Main dataset rows: {len(df):,}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

if "commit_id" not in df.columns:
    raise ValueError("Main dataset does not contain 'commit_id'.")

if "buggy" not in df.columns:
    raise ValueError("Main dataset does not contain 'buggy'.")


# ============================================================
# CLEAN COMMIT IDs
# ============================================================

szz["commit_id"] = szz["commit_id"].astype(str).str.strip()
df["commit_id"] = df["commit_id"].astype(str).str.strip()


# ============================================================
# CHECK DUPLICATES
# ============================================================

if szz["commit_id"].duplicated().any():
    raise ValueError("SZZ dataset contains duplicate commit_id values.")

if df["commit_id"].duplicated().any():
    raise ValueError("Main dataset contains duplicate commit_id values.")


# ============================================================
# BUILD SZZ LOOKUP
# ============================================================

szz_lookup = szz.set_index("commit_id")["buggy"]


# ============================================================
# FIND MATCHES
# ============================================================

matching_mask = df["commit_id"].isin(szz_lookup.index)

matching_count = matching_mask.sum()
unmatched_count = (~matching_mask).sum()


print("\n" + "=" * 75)
print("MATCHING RESULTS")
print("=" * 75)

print(f"Main dataset rows       : {len(df):,}")
print(f"Matching commit IDs     : {matching_count:,}")
print(f"Unmatched commit IDs    : {unmatched_count:,}")

print(
    f"Match percentage        : "
    f"{matching_count / len(df) * 100:.2f}%"
)


# ============================================================
# SAFETY CHECK
# ============================================================

expected_matches = len(szz)

if matching_count != expected_matches:
    raise ValueError(
        f"\nSAFETY CHECK FAILED!\n"
        f"Expected {expected_matches:,} matching commits "
        f"but found {matching_count:,}."
    )

print("\nSafety check passed.")
print(f"All {expected_matches:,} SZZ commits have matching commit_id values.")


# ============================================================
# SAVE ORIGINAL BUGGY VALUES FOR COMPARISON
# ============================================================

original_buggy = df["buggy"].copy()


# ============================================================
# REPLACE BUGGY ONLY FOR MATCHING COMMIT IDs
# ============================================================

df.loc[matching_mask, "buggy"] = (
    df.loc[matching_mask, "commit_id"]
    .map(szz_lookup)
)


# ============================================================
# VERIFY UNMATCHED ROWS WERE NOT CHANGED
# ============================================================

unchanged_unmatched = (
    df.loc[~matching_mask, "buggy"].equals(
        original_buggy.loc[~matching_mask]
    )
)

if not unchanged_unmatched:
    raise ValueError(
        "SAFETY CHECK FAILED: Unmatched rows were modified."
    )

print("Unmatched rows verified unchanged.")


# ============================================================
# VERIFY MATCHED VALUES
# ============================================================

expected_buggy = df.loc[matching_mask, "commit_id"].map(szz_lookup)

matched_correct = (
    df.loc[matching_mask, "buggy"].reset_index(drop=True)
    == expected_buggy.reset_index(drop=True)
).all()

if not matched_correct:
    raise ValueError(
        "SAFETY CHECK FAILED: Some matched buggy values are incorrect."
    )

print("All matched buggy values verified correctly.")


# ============================================================
# BUGGY STATISTICS
# ============================================================

print("\n" + "=" * 75)
print("BUGGY STATISTICS AFTER REPLACEMENT")
print("=" * 75)

print("\nMain dataset BEFORE replacement:")
print(
    original_buggy
    .astype(str)
    .value_counts(dropna=False)
)

print("\nMain dataset AFTER replacement:")
print(
    df["buggy"]
    .astype(str)
    .value_counts(dropna=False)
)


# ============================================================
# SAVE NEW DATASET
# ============================================================

print("\n" + "=" * 75)
print("SAVING")
print("=" * 75)

df.to_csv(OUTPUT_FILE, index=False)

print(f"\nUpdated dataset saved to:")
print(OUTPUT_FILE)


# ============================================================
# FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 75)
print("FINAL VERIFICATION")
print("=" * 75)

output_df = pd.read_csv(OUTPUT_FILE)

print(f"Original rows : {len(df):,}")
print(f"Output rows   : {len(output_df):,}")
print(f"Matched rows  : {matching_count:,}")
print(f"Unmatched rows: {unmatched_count:,}")

print("\nOriginal dataset was NOT modified.")
print("Backup exists.")
print("Only the 'buggy' column was replaced for matching commit IDs.")

print("\nDONE.")
print("=" * 75)