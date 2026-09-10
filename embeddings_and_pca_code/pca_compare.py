"""
Train-only PCA component comparison.

This script is for selecting/justifying a PCA dimensionality BEFORE the final
PCA transformation.

IMPORTANT:
- Only chronological TRAIN embeddings are used.
- No validation/test embeddings are loaded.
- No random train/test split is performed.
- The comparison does not modify any dataset.
- It compares 256/384/512/768 (or user-specified values).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def parse_embedding(value) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=np.float32)


def load_train_sample(
    train_path: Path,
    sample_size: int | None,
    random_state: int,
) -> np.ndarray:
    """
    Load embeddings ONLY from train.csv.

    If sample_size is specified and smaller than the train set, use a
    deterministic random sample from TRAIN ONLY. This is not a validation/test
    split and cannot leak information from either of them.
    """
    df = pd.read_csv(train_path, usecols=["embedding"], dtype=str)

    if df.empty:
        raise ValueError(f"No embeddings found in {train_path}.")

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state)

    X = np.vstack([parse_embedding(v) for v in df["embedding"]]).astype(
        np.float32, copy=False
    )

    if X.shape[1] != 768:
        raise ValueError(f"Expected 768 dimensions, found {X.shape[1]}.")

    return X


def evaluate_components(
    X: np.ndarray,
    component_list: list[int],
    random_state: int,
) -> pd.DataFrame:
    rows = []

    for n_components in component_list:
        if n_components > min(X.shape):
            rows.append(
                {
                    "n_components": n_components,
                    "explained_variance_ratio": np.nan,
                    "reconstruction_mse": np.nan,
                    "fit_seconds": np.nan,
                    "status": f"skipped: requires <= {min(X.shape)}",
                }
            )
            continue

        start = time.perf_counter()

        pca = PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=random_state,
        )
        transformed = pca.fit_transform(X)
        reconstructed = pca.inverse_transform(transformed)

        fit_seconds = time.perf_counter() - start
        mse = float(np.mean((X - reconstructed) ** 2))
        explained = float(np.sum(pca.explained_variance_ratio_))

        rows.append(
            {
                "n_components": n_components,
                "explained_variance_ratio": explained,
                "explained_variance_percent": explained * 100.0,
                "reconstruction_mse": mse,
                "fit_seconds": fit_seconds,
                "compression_ratio": 768 / n_components,
                "status": "ok",
            }
        )

        print(
            f"{n_components:>4} components | "
            f"variance={explained * 100:.4f}% | "
            f"MSE={mse:.8f} | "
            f"fit={fit_seconds:.2f}s"
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare PCA dimensions using TRAIN embeddings only."
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("results/chronological_splits/train.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/pca_comparison"),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional number of TRAIN rows to use. Default: all training rows.",
    )
    parser.add_argument(
        "--components",
        type=str,
        default="256,384,512,768",
        help="Comma-separated component counts.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    if not args.train.exists():
        raise FileNotFoundError(f"Training split not found: {args.train}")

    component_list = [
        int(x.strip()) for x in args.components.split(",") if x.strip()
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("TRAIN-ONLY PCA COMPONENT COMPARISON")
    print("=" * 78)
    print(f"Source: {args.train}")
    print(f"Components: {component_list}")
    print(f"Sample size: {args.sample_size or 'ALL TRAIN ROWS'}")
    print()
    print("Validation embeddings: NOT USED")
    print("Test embeddings:       NOT USED")
    print()

    X = load_train_sample(args.train, args.sample_size, args.random_state)

    print(f"Loaded TRAIN embedding matrix: {X.shape}")
    print()

    results = evaluate_components(X, component_list, args.random_state)

    csv_path = args.output_dir / "pca_comparison_results.csv"
    results.to_csv(csv_path, index=False)

    # Small JSON summary for reproducibility.
    summary = {
        "source": str(args.train),
        "source_split": "train",
        "validation_used": False,
        "test_used": False,
        "sample_rows": int(len(X)),
        "input_dimensions": int(X.shape[1]),
        "components": component_list,
        "random_state": args.random_state,
        "results": results.replace({np.nan: None}).to_dict(orient="records"),
    }

    with open(args.output_dir / "pca_comparison_metadata.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Choose the smallest model retaining at least 95% variance, if one exists.
    valid = results[results["status"] == "ok"].copy()
    threshold_hits = valid[
        valid["explained_variance_ratio"] >= 0.95
    ].sort_values("n_components")

    if not threshold_hits.empty:
        recommendation = int(threshold_hits.iloc[0]["n_components"])
        recommendation_text = (
            f"Smallest tested dimensionality retaining >=95% variance: "
            f"{recommendation}"
        )
    else:
        recommendation_text = (
            "No tested dimensionality retained >=95% variance. "
            "Choose based on downstream validation performance and the "
            "variance/reconstruction trade-off."
        )

    with open(args.output_dir / "pca_comparison_report.txt", "w") as f:
        f.write("TRAIN-ONLY PCA COMPONENT COMPARISON\n")
        f.write("=" * 78 + "\n")
        f.write(f"Training rows used: {len(X):,}\n")
        f.write(f"Input dimensions: {X.shape[1]}\n")
        f.write("Validation used: NO\n")
        f.write("Test used: NO\n\n")
        f.write(results.to_string(index=False))
        f.write("\n\n")
        f.write(recommendation_text + "\n")

    print()
    print("=" * 78)
    print("COMPARISON COMPLETE")
    print("=" * 78)
    print(results.to_string(index=False))
    print()
    print(recommendation_text)
    print(f"Saved: {csv_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
