"""
Leakage-safe PCA pipeline for CodeBERT embeddings.

Methodology:
1. Read ONLY results/chronological_splits/train.csv to FIT PCA.
2. Fit PCA on ALL available training embeddings (default: 384 components).
3. Transform train, validation, and test using the SAME fitted PCA.
4. Preserve every non-embedding column exactly as read.
5. Do not shuffle, resample, normalize, or fit PCA on validation/test.

Expected input:
    results/chronological_splits/
        train.csv
        validation.csv
        test.csv

Outputs:
    results/pca/
        train_pca384.csv
        validation_pca384.csv
        test_pca384.csv
        pca_384_model.npz
        pca_metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def parse_embedding(value) -> np.ndarray:
    if isinstance(value, str):
        return np.asarray(json.loads(value), dtype=np.float32)
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=np.float32)
    raise ValueError(f"Unsupported embedding cell type: {type(value)}")


def load_all_train_embeddings(csv_path: Path) -> np.ndarray:
    """Load ONLY the training embedding matrix for PCA fitting."""
    df = pd.read_csv(csv_path, usecols=["embedding"], dtype=str)
    if df.empty:
        raise ValueError(f"No rows found in {csv_path}.")

    embeddings = [parse_embedding(v) for v in df["embedding"]]
    X = np.vstack(embeddings).astype(np.float32, copy=False)

    if X.ndim != 2:
        raise ValueError(f"Expected 2-D embeddings, got shape {X.shape}.")
    if X.shape[1] != 768:
        raise ValueError(
            f"Expected 768-D CodeBERT embeddings before PCA, got {X.shape[1]} dimensions."
        )
    if not np.isfinite(X).all():
        raise ValueError("Training embeddings contain NaN or infinite values.")

    return X


def fit_pca(X_train: np.ndarray, n_components: int, random_state: int) -> PCA:
    if n_components <= 0 or n_components > min(X_train.shape):
        raise ValueError(
            f"n_components={n_components} is invalid for training matrix {X_train.shape}."
        )

    print(
        f"Fitting PCA on TRAIN ONLY: {X_train.shape[0]:,} rows x "
        f"{X_train.shape[1]} dimensions..."
    )

    # Randomized SVD is substantially faster than a full SVD for this matrix
    # while remaining deterministic with a fixed random_state.
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=random_state,
    )
    pca.fit(X_train)
    return pca


def save_pca_model(model_path: Path, pca: PCA) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_path,
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        explained_variance=pca.explained_variance_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        singular_values=pca.singular_values_.astype(np.float32),
        n_components=np.array([pca.n_components_], dtype=np.int32),
    )


def serialize_embedding(vector: np.ndarray) -> str:
    return json.dumps(vector.astype(float).tolist(), separators=(",", ":"))


def transform_csv(
    input_path: Path,
    output_path: Path,
    pca: PCA,
    chunk_size: int,
    overwrite: bool,
) -> int:
    """Transform one split using the already-fitted TRAIN PCA."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Use --overwrite to replace it."
        )

    if overwrite and output_path.exists():
        output_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    reader = pd.read_csv(input_path, chunksize=chunk_size)
    total_rows = 0
    header_written = False

    for chunk_index, chunk in enumerate(reader):
        if "embedding" not in chunk.columns:
            raise ValueError(f"'embedding' column missing from {input_path}.")

        embeddings = np.vstack(
            [parse_embedding(v) for v in chunk["embedding"]]
        ).astype(np.float32, copy=False)

        if embeddings.shape[1] != 768:
            raise ValueError(
                f"{input_path}: expected 768-D embeddings, got {embeddings.shape[1]}."
            )

        reduced = pca.transform(embeddings).astype(np.float32, copy=False)

        # Only this column changes. Every other source column remains untouched.
        chunk["embedding"] = [
            serialize_embedding(vector) for vector in reduced
        ]

        chunk.to_csv(
            output_path,
            mode="a",
            header=not header_written,
            index=False,
            quoting=csv.QUOTE_MINIMAL,
        )
        header_written = True

        total_rows += len(chunk)
        print(
            f"{input_path.name}: completed chunk {chunk_index}; "
            f"wrote {len(chunk):,} rows (total {total_rows:,})."
        )

    if total_rows == 0:
        raise ValueError(f"No rows were written for {input_path}.")

    return total_rows


def verify_output(path: Path, expected_rows: int, expected_components: int) -> None:
    """Basic post-write validation without loading the whole output."""
    header = list(pd.read_csv(path, nrows=0).columns)
    if "embedding" not in header:
        raise ValueError(f"{path} has no embedding column.")

    count = 0
    first_dim = None

    for chunk in pd.read_csv(path, usecols=["embedding"], chunksize=2000, dtype=str):
        count += len(chunk)
        if first_dim is None and len(chunk):
            first_dim = len(parse_embedding(chunk.iloc[0]["embedding"]))

    if count != expected_rows:
        raise ValueError(
            f"{path}: expected {expected_rows:,} rows, found {count:,}."
        )

    if first_dim != expected_components:
        raise ValueError(
            f"{path}: expected {expected_components}-D embeddings, found {first_dim}-D."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit PCA on chronological TRAIN embeddings only and transform all splits."
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("results/chronological_splits"),
        help="Directory containing train.csv, validation.csv, and test.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/pca"),
        help="Directory for PCA-transformed datasets and model.",
    )
    parser.add_argument(
        "--components",
        type=int,
        default=384,
        help="Number of PCA components.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Rows per chunk when transforming CSV files.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used by randomized SVD.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PCA outputs.",
    )
    args = parser.parse_args()

    train_path = args.split_dir / "train.csv"
    validation_path = args.split_dir / "validation.csv"
    test_path = args.split_dir / "test.csv"

    for path in (train_path, validation_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Required split not found: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"pca_{args.components}_model.npz"

    print("=" * 78)
    print("LEAKAGE-SAFE CODEBERT PCA")
    print("=" * 78)
    print(f"Train:      {train_path}")
    print(f"Validation: {validation_path}")
    print(f"Test:       {test_path}")
    print(f"Components: {args.components}")
    print()
    print("IMPORTANT: PCA will be FIT ON TRAIN ONLY.")
    print("Validation and test will only be TRANSFORMED.")
    print()

    # ---- FIT: TRAIN ONLY ----
    X_train = load_all_train_embeddings(train_path)
    pca = fit_pca(X_train, args.components, args.random_state)

    explained = float(np.sum(pca.explained_variance_ratio_))
    print(f"PCA fit complete.")
    print(f"Training matrix: {X_train.shape}")
    print(f"Components: {pca.n_components_}")
    print(f"Explained variance retained: {explained * 100:.4f}%")

    save_pca_model(model_path, pca)
    print(f"Saved PCA model: {model_path}")

    # ---- TRANSFORM: SAME TRAIN-FITTED PCA ----
    output_paths = {
        "train": args.output_dir / f"train_pca{args.components}.csv",
        "validation": args.output_dir / f"validation_pca{args.components}.csv",
        "test": args.output_dir / f"test_pca{args.components}.csv",
    }

    expected_rows = {}
    for name, input_path in (
        ("train", train_path),
        ("validation", validation_path),
        ("test", test_path),
    ):
        # Count logical CSV records, not physical lines.
        # This correctly handles quoted fields containing newlines.
        row_count = 0

        reader = pd.read_csv(
            input_path,
            usecols=["commit_id"],
            dtype=str,
            chunksize=args.chunk_size,
        )

        for chunk in reader:
            row_count += len(chunk)

        expected_rows[name] = row_count
        print(f"{name}: detected {row_count:,} logical CSV rows")

    for name, input_path in (
        ("train", train_path),
        ("validation", validation_path),
        ("test", test_path),
    ):
        output_path = output_paths[name]
        print()
        print(f"Transforming {name.upper()} using the TRAIN-FITTED PCA...")
        written = transform_csv(
            input_path,
            output_path,
            pca,
            args.chunk_size,
            args.overwrite,
        )
        if written != expected_rows[name]:
            raise ValueError(
                f"{name}: input rows={expected_rows[name]:,}, output rows={written:,}"
            )
        verify_output(output_path, written, args.components)
        print(f"{name.upper()} verification: PASSED")

    metadata = {
        "method": "PCA",
        "input_embedding_dimension": 768,
        "output_embedding_dimension": args.components,
        "fit_source": "train.csv ONLY",
        "validation_used_for_fit": False,
        "test_used_for_fit": False,
        "random_state": args.random_state,
        "solver": "randomized",
        "training_rows": expected_rows["train"],
        "validation_rows": expected_rows["validation"],
        "test_rows": expected_rows["test"],
        "explained_variance_ratio": explained,
        "explained_variance_percent": explained * 100.0,
        "pca_model": str(model_path),
        "outputs": {k: str(v) for k, v in output_paths.items()},
    }

    with open(args.output_dir / "pca_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print()
    print("=" * 78)
    print("PCA COMPLETE")
    print("=" * 78)
    print(f"Train:      {output_paths['train']}")
    print(f"Validation: {output_paths['validation']}")
    print(f"Test:       {output_paths['test']}")
    print(f"Explained variance: {explained * 100:.4f}%")
    print("PCA was fitted on TRAIN ONLY: YES")
    print("=" * 78)


if __name__ == "__main__":
    main()
