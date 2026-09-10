from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import pandas as pd


def embedding_dim(value: str) -> int:
    vector = json.loads(value)
    if not isinstance(vector, list):
        raise ValueError("Embedding is not a JSON list.")
    return len(vector)


def load_split(path: Path, chunk_size: int) -> dict[str, str]:
    result = {}
    for chunk in pd.read_csv(path, usecols=["commit_id", "embedding"],
                             dtype=str, keep_default_na=False,
                             chunksize=chunk_size):
        for cid, emb in zip(chunk["commit_id"], chunk["embedding"]):
            if cid in result:
                raise ValueError(f"Duplicate commit_id in split: {cid}")
            result[cid] = emb
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Merge leakage-safe 384-D PCA embeddings into the original dataset."
    )
    parser.add_argument("--input", type=Path,
                        default=Path("data/final_multilanguage_59996_szz_buggy.csv"))
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("results/pca"),
    )
    parser.add_argument("--output", type=Path,
                        default=Path("data/final_multilanguage_szz_buggy_384d.csv"))
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} already exists; use --overwrite.")

    split_files = {
        "train": args.split_dir / "train_pca384.csv",
        "validation": args.split_dir / "validation_pca384.csv",
        "test": args.split_dir / "test_pca384.csv",
    }
    for name, path in split_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} split: {path}")

    print("=" * 78)
    print("MERGING LEAKAGE-SAFE 384-D PCA EMBEDDINGS")
    print("=" * 78)

    pca_embeddings = {}
    for name, path in split_files.items():
        current = load_split(path, args.chunk_size)
        overlap = set(pca_embeddings).intersection(current)
        if overlap:
            raise ValueError(f"Commit overlap between PCA splits: {next(iter(overlap))}")
        pca_embeddings.update(current)
        print(f"{name}: {len(current):,} embeddings")

    if len(pca_embeddings) != 59996:
        raise ValueError(f"Expected 59,996 PCA embeddings; found {len(pca_embeddings):,}.")

    dims = set(embedding_dim(v) for v in pca_embeddings.values())
    if dims != {384}:
        raise ValueError(f"Expected every embedding to be 384-D; found dimensions {dims}.")
    print("Verified: all PCA embeddings are 384-D.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    original_columns = None
    input_ids = set()
    total = 0

    reader = pd.read_csv(args.input, dtype=str, keep_default_na=False,
                          chunksize=args.chunk_size)

    for i, chunk in enumerate(reader):
        if original_columns is None:
            original_columns = list(chunk.columns)
        if "commit_id" not in chunk.columns or "embedding" not in chunk.columns:
            raise ValueError("Input must contain commit_id and embedding columns.")

        input_ids.update(chunk["commit_id"].tolist())
        missing = [cid for cid in chunk["commit_id"] if cid not in pca_embeddings]
        if missing:
            raise ValueError(f"Missing PCA embedding for commit {missing[0]}.")

        # Replace ONLY embedding; preserve original row order and all other columns.
        chunk["embedding"] = [pca_embeddings[cid] for cid in chunk["commit_id"]]

        chunk.to_csv(args.output, mode="w" if i == 0 else "a",
                     header=(i == 0), index=False, quoting=csv.QUOTE_MINIMAL)
        total += len(chunk)
        print(f"Completed chunk {i}: {len(chunk):,} rows (total {total:,}).")

    if total != 59996:
        raise ValueError(f"Final row count {total:,} != 59,996.")
    if len(input_ids) != 59996:
        raise ValueError(f"Input unique commit IDs {len(input_ids):,} != 59,996.")

    final_header = list(pd.read_csv(args.output, nrows=0).columns)
    if final_header != original_columns:
        raise ValueError("Final columns/order differ from original input.")

    print()
    print("=" * 78)
    print("FINAL DATASET CREATED SUCCESSFULLY")
    print("=" * 78)
    print(f"Rows:                {total:,}")
    print(f"Unique commit IDs:   {len(input_ids):,}")
    print("Embedding dimension: 384")
    print("Original row order:  PRESERVED")
    print("Original columns:    PRESERVED")
    print("PCA fitting:         TRAIN ONLY")
    print(f"Output:              {args.output}")
    print("=" * 78)


if __name__ == "__main__":
    main()
