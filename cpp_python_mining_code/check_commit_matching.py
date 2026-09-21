import pandas as pd

# ============================================================
# FILE PATHS
# ============================================================

szz_file = r"D:\embeddings_and_pca_code\cpp_python_mining_code\python_cpp_23996_szz.csv"

main_file = r"D:\embeddings_and_pca_code\data\final_multilanguage_59996_768d_reconstructed.csv"


# ============================================================
# READ ONLY commit_id COLUMNS
# ============================================================

print("=" * 70)
print("COMMIT ID MATCHING ANALYSIS")
print("=" * 70)

print("\nReading SZZ dataset...")
szz = pd.read_csv(szz_file, usecols=["commit_id"])

print("Reading 59996 dataset...")
main = pd.read_csv(main_file, usecols=["commit_id"])


# ============================================================
# CLEAN COMMIT IDs
# ============================================================

szz_ids = szz["commit_id"].astype(str).str.strip()
main_ids = main["commit_id"].astype(str).str.strip()


# ============================================================
# BASIC ROW COUNTS
# ============================================================

print("\n" + "=" * 70)
print("1. ROW COUNTS")
print("=" * 70)

print(f"SZZ dataset rows       : {len(szz_ids):,}")
print(f"59996 dataset rows     : {len(main_ids):,}")


# ============================================================
# UNIQUE COMMIT IDs
# ============================================================

szz_unique = set(szz_ids)
main_unique = set(main_ids)

print("\n" + "=" * 70)
print("2. UNIQUE COMMIT IDs")
print("=" * 70)

print(f"SZZ unique commit IDs      : {len(szz_unique):,}")
print(f"59996 unique commit IDs    : {len(main_unique):,}")


# ============================================================
# DUPLICATES
# ============================================================

szz_duplicates = len(szz_ids) - len(szz_unique)
main_duplicates = len(main_ids) - len(main_unique)

print("\n" + "=" * 70)
print("3. DUPLICATE COMMIT IDs")
print("=" * 70)

print(f"SZZ duplicate rows         : {szz_duplicates:,}")
print(f"59996 duplicate rows       : {main_duplicates:,}")


# ============================================================
# MATCHING COMMIT IDs
# ============================================================

matching_ids = szz_unique & main_unique

szz_only = szz_unique - main_unique
main_only = main_unique - szz_unique


print("\n" + "=" * 70)
print("4. MATCHING COMMIT IDs")
print("=" * 70)

print(f"Matching unique commit IDs : {len(matching_ids):,}")

print(f"SZZ IDs NOT in 59996       : {len(szz_only):,}")
print(f"59996 IDs NOT in SZZ       : {len(main_only):,}")


# ============================================================
# MATCHING PERCENTAGES
# ============================================================

szz_match_percentage = (
    len(matching_ids) / len(szz_unique) * 100
    if len(szz_unique) > 0 else 0
)

main_match_percentage = (
    len(matching_ids) / len(main_unique) * 100
    if len(main_unique) > 0 else 0
)


print("\n" + "=" * 70)
print("5. MATCHING PERCENTAGES")
print("=" * 70)

print(
    f"Percentage of SZZ IDs found in 59996 : "
    f"{szz_match_percentage:.2f}%"
)

print(
    f"Percentage of 59996 IDs found in SZZ : "
    f"{main_match_percentage:.2f}%"
)


# ============================================================
# ROW-LEVEL MATCHING
# ============================================================

szz_match_rows = szz_ids.isin(main_unique).sum()
main_match_rows = main_ids.isin(szz_unique).sum()

print("\n" + "=" * 70)
print("6. ROW-LEVEL MATCHING")
print("=" * 70)

print(
    f"SZZ rows whose commit_id exists in 59996 : "
    f"{szz_match_rows:,}"
)

print(
    f"59996 rows whose commit_id exists in SZZ : "
    f"{main_match_rows:,}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"SZZ rows                    : {len(szz_ids):,}")
print(f"59996 rows                  : {len(main_ids):,}")
print(f"SZZ unique IDs              : {len(szz_unique):,}")
print(f"59996 unique IDs            : {len(main_unique):,}")
print(f"MATCHING unique IDs         : {len(matching_ids):,}")
print(f"SZZ IDs missing in 59996   : {len(szz_only):,}")
print(f"59996 IDs missing in SZZ   : {len(main_only):,}")
print(f"SZZ → 59996 match rate      : {szz_match_percentage:.2f}%")
print(f"59996 → SZZ match rate      : {main_match_percentage:.2f}%")

print("\nNo files were modified.")
print("No columns were replaced.")
print("=" * 70)