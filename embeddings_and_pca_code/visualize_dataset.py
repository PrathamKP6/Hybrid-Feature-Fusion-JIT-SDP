"""
Leakage-safe visualization utility for the CodeBERT embedding splits.

This script is VISUALIZATION ONLY. It must not be used to create the PCA
features used by the ML experiments.

Key rule:
- The 2-D PCA visualization is fitted on TRAIN ONLY and then applied to the
  selected data. This avoids fitting the visualization PCA on test data.
- t-SNE/UMAP are exploratory visualizations and are clearly labeled as such.
- No output dataset is modified.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_embedding(value) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=np.float32)


def load_embeddings(csv_path: Path, max_rows: int | None = None):
    df = pd.read_csv(csv_path, nrows=max_rows, dtype=str)

    if "embedding" not in df.columns:
        raise ValueError(f"'embedding' column missing from {csv_path}.")

    embeddings = np.vstack(
        [parse_embedding(v) for v in df["embedding"]]
    ).astype(np.float32, copy=False)

    return df, embeddings


def sample_indices(n: int, sample_size: int, seed: int) -> np.ndarray:
    if n <= sample_size:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=sample_size, replace=False)


def plot_embedding_statistics(
    embeddings: np.ndarray,
    output_dir: Path,
    prefix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    variances = embeddings.var(axis=0)
    norms = np.linalg.norm(embeddings, axis=1)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.hist(variances, bins=50)
    ax.set_title(f"Embedding Dimension Variance — {prefix}")
    ax.set_xlabel("Variance")
    ax.set_ylabel("Number of dimensions")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_dimension_variance.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.hist(norms, bins=50)
    ax.set_title(f"Embedding L2 Norm Distribution — {prefix}")
    ax.set_xlabel("L2 norm")
    ax.set_ylabel("Number of embeddings")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_embedding_norms.png", dpi=150)
    plt.close(fig)


def plot_train_fitted_pca_2d(
    train_embeddings: np.ndarray,
    selected_embeddings: np.ndarray,
    output_path: Path,
    selected_label: str,
    seed: int,
) -> None:
    """
    Fit 2-D PCA on TRAIN ONLY.

    The selected split is only transformed by the already-fitted PCA.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, svd_solver="randomized", random_state=seed)
    pca.fit(train_embeddings)

    projected = pca.transform(selected_embeddings)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.scatter(projected[:, 0], projected[:, 1], s=12, alpha=0.5)

    explained = float(np.sum(pca.explained_variance_ratio_))
    ax.set_title(
        f"Train-Fitted PCA 2-D Visualization — {selected_label} "
        f"(explained variance: {explained * 100:.2f}%)"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_tsne(
    embeddings: np.ndarray,
    output_path: Path,
    label: str,
    seed: int,
) -> None:
    from sklearn.manifold import TSNE

    tsne = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=min(30, max(5, (len(embeddings) - 1) // 3)),
        max_iter=1000,
    )
    projected = tsne.fit_transform(embeddings)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.scatter(projected[:, 0], projected[:, 1], s=12, alpha=0.5)
    ax.set_title(f"Exploratory t-SNE — {label} (not an experimental feature)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_umap(
    embeddings: np.ndarray,
    output_path: Path,
    label: str,
    seed: int,
) -> None:
    try:
        import umap
    except ImportError:
        print("umap-learn is not installed; skipping UMAP.")
        return

    reducer = umap.UMAP(
        n_components=2,
        random_state=seed,
        n_neighbors=min(15, max(2, len(embeddings) - 1)),
        min_dist=0.1,
    )
    projected = reducer.fit_transform(embeddings)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.scatter(projected[:, 0], projected[:, 1], s=12, alpha=0.5)
    ax.set_title(f"Exploratory UMAP — {label} (not an experimental feature)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize CodeBERT embeddings without creating experimental features."
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("results/chronological_splits/train.csv"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("results/chronological_splits/validation.csv"),
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("results/chronological_splits/test.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/visualizations"),
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="train",
        help="Split to visualize with train-fitted PCA.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Maximum rows used for visualization.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-tsne", action="store_true")
    parser.add_argument("--skip-umap", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": args.train,
        "validation": args.validation,
        "test": args.test,
    }

    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing split: {path}")

    # Train embeddings are always loaded for fitting the visualization PCA.
    train_df, train_embeddings = load_embeddings(args.train)

    selected_df, selected_embeddings = load_embeddings(paths[args.split])

    train_idx = sample_indices(
        len(train_embeddings), args.sample_size, args.seed
    )
    selected_idx = sample_indices(
        len(selected_embeddings), args.sample_size, args.seed
    )

    train_sample = train_embeddings[train_idx]
    selected_sample = selected_embeddings[selected_idx]

    print("=" * 78)
    print("LEAKAGE-SAFE EMBEDDING VISUALIZATION")
    print("=" * 78)
    print(f"Train rows loaded: {len(train_embeddings):,}")
    print(f"Selected split: {args.split}")
    print(f"Selected rows loaded: {len(selected_embeddings):,}")
    print()
    print("2-D PCA visualization is FIT ON TRAIN ONLY.")
    print("This script does NOT generate the PCA features used by experiments.")
    print()

    plot_embedding_statistics(
        selected_sample,
        args.output_dir,
        prefix=args.split,
    )

    plot_train_fitted_pca_2d(
        train_sample,
        selected_sample,
        args.output_dir / f"{args.split}_train_fitted_pca_2d.png",
        args.split,
        args.seed,
    )

    if not args.skip_tsne:
        plot_tsne(
            selected_sample,
            args.output_dir / f"{args.split}_tsne.png",
            args.split,
            args.seed,
        )

    if not args.skip_umap:
        plot_umap(
            selected_sample,
            args.output_dir / f"{args.split}_umap.png",
            args.split,
            args.seed,
        )

    print()
    print(f"Visualizations saved to: {args.output_dir}")
    print("No experimental feature dataset was created.")


if __name__ == "__main__":
    main()
