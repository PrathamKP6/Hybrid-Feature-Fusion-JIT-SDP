import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent.parent / "data"

CHRONO_DIR = BASE / "chronological_split_dataset"
LLM_FILE = BASE / "llm_features.csv"
OUTPUT_DIR = BASE / "llm_experiment_dataset"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SPLIT FILES
# ============================================================

splits = {
    "train": CHRONO_DIR / "train.csv",
    "validation": CHRONO_DIR / "validation.csv",
    "test": CHRONO_DIR / "test.csv"
}


# ============================================================
# START
# ============================================================

print("=" * 70)
print("CREATING LLM EXPERIMENT DATASET")
print("=" * 70)


# ============================================================
# READ LLM FEATURES
# ============================================================

print("\nReading llm_features.csv...")

llm_df = pd.read_csv(
    LLM_FILE,
    dtype={"commit_id": str},
    low_memory=False
)

print(f"LLM dataset shape: {llm_df.shape}")


# ============================================================
# CHECK COMMIT ID
# ============================================================

if "commit_id" not in llm_df.columns:
    raise ValueError(
        "ERROR: commit_id column not found in llm_features.csv"
    )


# ============================================================
# CHECK DUPLICATE COMMIT IDS
# ============================================================

print("\nChecking duplicate commit IDs...")

duplicate_mask = llm_df["commit_id"].duplicated(keep=False)

duplicate_ids = (
    llm_df.loc[duplicate_mask, "commit_id"]
    .dropna()
    .unique()
)

if len(duplicate_ids) == 0:

    print("OK: No duplicate commit IDs found.")

else:

    print(
        f"WARNING: {len(duplicate_ids)} duplicate commit IDs found!"
    )

    duplicate_file = OUTPUT_DIR / "duplicate_commit_ids.txt"

    with open(duplicate_file, "w", encoding="utf-8") as f:

        for commit_id in duplicate_ids:
            f.write(str(commit_id) + "\n")

    print(f"Duplicate IDs saved to:")
    print(duplicate_file)


# ============================================================
# CREATE LOOKUP
# ============================================================

llm_lookup = llm_df.set_index("commit_id", drop=False)


# ============================================================
# PROCESS TRAIN / VALIDATION / TEST
# ============================================================

total_missing = 0


for split_name, chrono_file in splits.items():

    print("\n" + "=" * 70)
    print(f"PROCESSING {split_name.upper()}")
    print("=" * 70)


    # --------------------------------------------------------
    # READ CHRONOLOGICAL SPLIT
    # --------------------------------------------------------

    chrono_df = pd.read_csv(
        chrono_file,
        dtype={"commit_id": str},
        low_memory=False
    )

    if "commit_id" not in chrono_df.columns:
        raise ValueError(
            f"ERROR: commit_id column not found in {chrono_file}"
        )

    print(
        f"Chronological {split_name} rows: "
        f"{len(chrono_df)}"
    )


    # --------------------------------------------------------
    # GET COMMIT IDS
    # --------------------------------------------------------

    chrono_ids = chrono_df["commit_id"].tolist()


    # --------------------------------------------------------
    # FIND MISSING IDS
    # --------------------------------------------------------

    missing_ids = [
        commit_id
        for commit_id in chrono_ids
        if commit_id not in llm_lookup.index
    ]


    if len(missing_ids) == 0:

        print(
            "OK: All chronological commit IDs "
            "exist in llm_features.csv."
        )

    else:

        print(
            f"WARNING: {len(missing_ids)} commit IDs "
            f"are missing from llm_features.csv."
        )

        missing_file = (
            OUTPUT_DIR /
            f"{split_name}_missing_commit_ids.txt"
        )

        with open(missing_file, "w", encoding="utf-8") as f:

            for commit_id in missing_ids:
                f.write(str(commit_id) + "\n")

        print(f"Missing IDs saved to:")
        print(missing_file)

        total_missing += len(missing_ids)


    # --------------------------------------------------------
    # KEEP ONLY MATCHING IDS
    # --------------------------------------------------------

    valid_ids = [
        commit_id
        for commit_id in chrono_ids
        if commit_id in llm_lookup.index
    ]


    # --------------------------------------------------------
    # SELECT ORIGINAL LLM ROWS
    # IN CHRONOLOGICAL ORDER
    # --------------------------------------------------------

    output_df = llm_lookup.loc[valid_ids].copy()


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_file = OUTPUT_DIR / f"{split_name}.csv"

    output_df.to_csv(
        output_file,
        index=False
    )

    print(f"\nCreated:")
    print(output_file)

    print(f"Output shape: {output_df.shape}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("COMPLETED")
print("=" * 70)

print(f"\nOutput folder:")
print(OUTPUT_DIR)

print("\nFiles created:")

for split_name in splits:

    output_file = OUTPUT_DIR / f"{split_name}.csv"

    if output_file.exists():

        df = pd.read_csv(
            output_file,
            dtype={"commit_id": str},
            low_memory=False
        )

        print(
            f"  {split_name}.csv : {df.shape}"
        )


print(f"\nTotal missing commit IDs: {total_missing}")

print("\nThe output files contain:")
print("- All original columns from llm_features.csv")
print("- All original feature values")
print("- Only rows selected using chronological commit IDs")
print("- Train/validation/test ordering follows chronological files")

print("\nNo PCA, scaling, encoding, feature modification,")
print("or column modification was performed.")

print("=" * 70)