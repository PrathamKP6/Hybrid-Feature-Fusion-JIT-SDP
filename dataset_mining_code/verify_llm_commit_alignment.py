from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

LLM_DIR = Path("data/llm_experiment_dataset")
BASE_DIR = Path("results/chronological_splits")

FILES = {
    "train": (
        BASE_DIR / "train.csv",
        LLM_DIR / "train.csv"
    ),
    "validation": (
        BASE_DIR / "validation.csv",
        LLM_DIR / "validation.csv"
    ),
    "test": (
        BASE_DIR / "test.csv",
        LLM_DIR / "test.csv"
    ),
}


# ============================================================
# LOAD
# ============================================================

def load_csv(path):
    print(f"Loading: {path}")
    return pd.read_csv(path)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("LLM COMMIT-ID ALIGNMENT VERIFICATION")
    print("=" * 90)

    base = {}
    llm = {}

    for split, (base_path, llm_path) in FILES.items():

        base[split] = load_csv(base_path)
        llm[split] = load_csv(llm_path)

        print(
            f"\n{split.upper()}: "
            f"base={len(base[split]):,}, "
            f"llm={len(llm[split]):,}"
        )

    # --------------------------------------------------------
    # Basic checks
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("BASIC COMMIT-ID CHECK")
    print("=" * 90)

    for split in ["train", "validation", "test"]:

        b = base[split]
        l = llm[split]

        if "commit_id" not in b.columns:
            print(f"[ERROR] Base {split} has no commit_id")
            continue

        if "commit_id" not in l.columns:
            print(f"[ERROR] LLM {split} has no commit_id")
            continue

        b_ids = set(b["commit_id"].astype(str))
        l_ids = set(l["commit_id"].astype(str))

        intersection = b_ids & l_ids

        base_only = b_ids - l_ids
        llm_only = l_ids - b_ids

        print(f"\n{split.upper()}")

        print(f"Base IDs       : {len(b_ids):,}")
        print(f"LLM IDs        : {len(l_ids):,}")
        print(f"Matching IDs   : {len(intersection):,}")
        print(f"Base-only IDs  : {len(base_only):,}")
        print(f"LLM-only IDs   : {len(llm_only):,}")

        if b_ids == l_ids:
            print("[OK] EXACT SAME COMMIT-ID SET")
        else:
            print("[WARNING] COMMIT-ID SETS DIFFER")

            if base_only:
                print("\nFirst 20 base-only IDs:")
                for x in list(base_only)[:20]:
                    print(" ", x)

            if llm_only:
                print("\nFirst 20 LLM-only IDs:")
                for x in list(llm_only)[:20]:
                    print(" ", x)

    # --------------------------------------------------------
    # Cross-split leakage
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("CROSS-SPLIT COMMIT-ID OVERLAP")
    print("=" * 90)

    base_sets = {
        split: set(df["commit_id"].astype(str))
        for split, df in base.items()
    }

    llm_sets = {
        split: set(df["commit_id"].astype(str))
        for split, df in llm.items()
    }

    splits = ["train", "validation", "test"]

    print("\nYOUR BASE DATASET:")

    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):

            a = splits[i]
            b = splits[j]

            overlap = base_sets[a] & base_sets[b]

            print(
                f"{a} ∩ {b}: {len(overlap):,}"
            )

    print("\nLLM DATASET:")

    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):

            a = splits[i]
            b = splits[j]

            overlap = llm_sets[a] & llm_sets[b]

            print(
                f"{a} ∩ {b}: {len(overlap):,}"
            )

    # --------------------------------------------------------
    # Important: determine where each LLM split's IDs belong
    # in the BASE splits
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("LLM SPLIT → BASE SPLIT MAPPING")
    print("=" * 90)

    for llm_split in splits:

        print(f"\nLLM {llm_split.upper()}:")

        ids = llm_sets[llm_split]

        for base_split in splits:

            overlap = ids & base_sets[base_split]

            print(
                f"  matches BASE {base_split:10s}: "
                f"{len(overlap):,}"
            )

    # --------------------------------------------------------
    # Row-order check for common IDs
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("ROW ORDER CHECK")
    print("=" * 90)

    for split in splits:

        b = base[split]
        l = llm[split]

        common = set(
            b["commit_id"].astype(str)
        ) & set(
            l["commit_id"].astype(str)
        )

        if len(common) == 0:
            print(f"{split}: no common IDs")
            continue

        # Compare order only for common IDs
        base_order = [
            x for x in b["commit_id"].astype(str)
            if x in common
        ]

        llm_order = [
            x for x in l["commit_id"].astype(str)
            if x in common
        ]

        same_order = base_order == llm_order

        print(
            f"{split.upper()}: "
            f"{'SAME ORDER' if same_order else 'DIFFERENT ORDER'} "
            f"({len(common):,} common IDs)"
        )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("VERIFICATION COMPLETE")
    print("=" * 90)

    print("""
DO NOT merge the datasets yet.

The important result is:

1. Do the commit-ID sets match?
2. Where do the 1-2 boundary differences occur?
3. Are the LLM IDs assigned to the same train/validation/test
   partitions as your canonical chronological split?
4. Is the row ordering identical?

Once this is established, we can safely reconstruct the
LLM + JIT + CodeBERT combinations.
""")


if __name__ == "__main__":
    main()