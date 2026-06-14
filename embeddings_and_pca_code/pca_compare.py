"""Sample-based PCA comparison for CodeBERT embeddings.

This script avoids loading the full dataset into memory by reservoir-sampling the
embedding column, then fits PCA models at 768, 512, 384, and 256 components and
compares them automatically using reconstruction error, explained variance, and
fit time.
"""

import argparse
import json
import os
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_embedding(value) -> np.ndarray:
    """Parse a single embedding cell from the CSV."""
    if isinstance(value, str):
        return np.asarray(json.loads(value), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def reservoir_sample_embeddings(
    csv_path: str,
    sample_size: int,
    chunksize: int = 5000,
    random_state: int = 42,
) -> Tuple[np.ndarray, int]:
    """Load only a bounded sample of embeddings from the CSV."""
    df = pd.read_csv(csv_path, usecols=["embedding"], nrows=sample_size)
    if df.empty:
        raise ValueError("No embeddings were loaded from the CSV file.")

    sample = [parse_embedding(value) for value in df["embedding"]]
    return np.vstack(sample), len(sample)


def fit_pca_svd(X_train: np.ndarray, n_components: int) -> dict:
    """Fit PCA using NumPy SVD and return the parameters needed for transform/inverse_transform."""
    mean_vector = X_train.mean(axis=0)
    centered = X_train - mean_vector

    # full_matrices=False keeps the decomposition compact and workable for sampled data.
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)

    total_variance = float(np.sum((singular_values ** 2) / max(1, X_train.shape[0] - 1)))
    explained_variance = (singular_values[:n_components] ** 2) / max(1, X_train.shape[0] - 1)
    components = vt[:n_components]

    return {
        "mean": mean_vector,
        "components": components,
        "explained_variance_ratio": float(explained_variance.sum() / total_variance) if total_variance > 0 else 0.0,
    }


def transform_pca(X: np.ndarray, mean_vector: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Project data onto the PCA subspace."""
    return (X - mean_vector) @ components.T


def inverse_transform_pca(X_transformed: np.ndarray, mean_vector: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Reconstruct data from the PCA subspace."""
    return X_transformed @ components + mean_vector


def evaluate_pca(X_train: np.ndarray, X_test: np.ndarray, n_components: int, random_state: int = 42) -> dict:
    """Fit PCA and compute comparison metrics."""
    start_time = time.perf_counter()
    pca = fit_pca_svd(X_train, n_components)
    fit_seconds = time.perf_counter() - start_time

    X_train_transformed = transform_pca(X_train, pca["mean"], pca["components"])
    X_test_transformed = transform_pca(X_test, pca["mean"], pca["components"])
    X_train_recon = inverse_transform_pca(X_train_transformed, pca["mean"], pca["components"])
    X_test_recon = inverse_transform_pca(X_test_transformed, pca["mean"], pca["components"])

    train_mse = float(np.mean((X_train - X_train_recon) ** 2))
    test_mse = float(np.mean((X_test - X_test_recon) ** 2))

    return {
        "n_components": int(n_components),
        "fit_seconds": float(fit_seconds),
        "explained_variance_ratio": float(pca["explained_variance_ratio"]),
        "train_reconstruction_mse": train_mse,
        "test_reconstruction_mse": test_mse,
        "compression_ratio": float(X_train.shape[1] / n_components),
    }


def compare_pca_components(
    embeddings: np.ndarray,
    component_list: List[int],
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Split the sample, fit PCA models, and compare them."""
    if embeddings.ndim != 2:
        raise ValueError("Expected a 2D embedding matrix.")

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(embeddings))
    split_idx = max(1, int(len(indices) * (1.0 - test_fraction)))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    if len(test_idx) == 0:
        raise ValueError("Test split is empty. Increase the sample size or reduce test_fraction.")

    X_train = embeddings[train_idx]
    X_test = embeddings[test_idx]

    results = []
    max_valid_components = min(X_train.shape[0], X_train.shape[1])
    for n_components in component_list:
        if n_components > max_valid_components:
            results.append({
                "n_components": int(n_components),
                "fit_seconds": np.nan,
                "explained_variance_ratio": np.nan,
                "train_reconstruction_mse": np.nan,
                "test_reconstruction_mse": np.nan,
                "compression_ratio": np.nan,
                "status": f"skipped: requires <= {max_valid_components} components for this sample",
            })
            continue

        metrics = evaluate_pca(X_train, X_test, n_components, random_state=random_state)
        metrics["status"] = "ok"
        results.append(metrics)

    return pd.DataFrame(results)


def save_report(
    results: pd.DataFrame,
    output_dir: str,
    source_rows: int,
    sample_rows: int,
    components: List[int],
    variance_threshold: float,
) -> None:
    """Write a text report and a CSV summary."""
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "pca_comparison_results.csv")
    results.to_csv(csv_path, index=False)

    valid = results[results["status"] == "ok"].copy()
    if valid.empty:
        recommendation = "No PCA run was valid for the sampled data. Increase sample size."
    else:
        threshold_hits = valid[valid["explained_variance_ratio"] >= variance_threshold].sort_values("n_components")
        if not threshold_hits.empty:
            chosen = threshold_hits.iloc[0]
            recommendation = (
                f"Smallest model meeting {variance_threshold * 100:.1f}% variance retention: "
                f"{int(chosen['n_components'])} components "
                f"({chosen['explained_variance_ratio'] * 100:.2f}% explained variance)."
            )
        else:
            best = valid.sort_values(["explained_variance_ratio", "n_components"], ascending=[False, True]).iloc[0]
            recommendation = (
                f"No run reached {variance_threshold * 100:.1f}% variance retention. Best in this sweep: "
                f"{int(best['n_components'])} components ({best['explained_variance_ratio'] * 100:.2f}% explained variance)."
            )

    lines = [
        "=============================================================================",
        "                         PCA COMPARISON REPORT",
        "=============================================================================",
        "",
        f"Source rows scanned: {source_rows:,}",
        f"Sample rows used: {sample_rows:,}",
        f"Component sweep: {', '.join(str(c) for c in components)}",
        f"Variance retention threshold: {variance_threshold * 100:.1f}%",
        "",
        "Comparison metrics:",
        results.to_string(index=False),
        "",
        "Interpretation:",
        "  - Lower reconstruction MSE is better.",
        "  - Higher explained variance is better.",
        "  - Lower fit time is better.",
        "",
        f"Recommendation: {recommendation}",
        "=============================================================================",
    ]

    report_path = os.path.join(output_dir, "pca_comparison_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved: {csv_path}")
    print(f"Saved: {report_path}")
    print("\n" + "\n".join(lines))


def plot_comparison(results: pd.DataFrame, output_dir: str) -> None:
    """Create a compact visual comparison of the PCA sweep."""
    os.makedirs(output_dir, exist_ok=True)

    valid = results[results["status"] == "ok"].copy()
    if valid.empty:
        return

    valid = valid.sort_values("n_components")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("PCA Component Sweep", fontsize=16, fontweight="bold")

    axes[0].plot(valid["n_components"], valid["explained_variance_ratio"] * 100, marker="o", linewidth=2)
    axes[0].set_title("Explained Variance")
    axes[0].set_xlabel("Components")
    axes[0].set_ylabel("Explained Variance (%)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(valid["n_components"], valid["test_reconstruction_mse"], marker="o", linewidth=2, color="darkorange")
    axes[1].set_title("Test Reconstruction Error")
    axes[1].set_xlabel("Components")
    axes[1].set_ylabel("MSE")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(valid["n_components"], valid["fit_seconds"], marker="o", linewidth=2, color="seagreen")
    axes[2].set_title("Fit Time")
    axes[2].set_xlabel("Components")
    axes[2].set_ylabel("Seconds")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "pca_comparison_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {plot_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample-based PCA comparison for embedding compression")
    parser.add_argument("--input", type=str, default="./final_multilanguage_dataset_with_embeddings.csv", help="Path to the CSV with embeddings")
    parser.add_argument("--output-dir", type=str, default="./visualizations", help="Directory for reports and plots")
    parser.add_argument("--sample-size", type=int, default=20000, help="Number of embeddings to sample from the CSV")
    parser.add_argument("--chunksize", type=int, default=5000, help="Unused compatibility argument; sampling is bounded by nrows")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Fraction of the sample held out for evaluation")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--variance-threshold", type=float, default=0.99, help="Retention threshold used for the automatic recommendation")
    parser.add_argument(
        "--components",
        type=str,
        default="768,512,384,256",
        help="Comma-separated PCA component sizes to compare",
    )

    args = parser.parse_args()

    component_list = [int(value.strip()) for value in args.components.split(",") if value.strip()]
    if not component_list:
        raise ValueError("No PCA component sizes were provided.")

    print(f"Sampling up to {args.sample_size:,} embeddings from {args.input}...")
    embeddings, source_rows = reservoir_sample_embeddings(
        args.input,
        sample_size=args.sample_size,
        chunksize=args.chunksize,
        random_state=args.random_state,
    )
    print(f"Loaded sample shape: {embeddings.shape} from {source_rows:,} source rows")

    results = compare_pca_components(
        embeddings,
        component_list=component_list,
        test_fraction=args.test_fraction,
        random_state=args.random_state,
    )

    save_report(results, args.output_dir, source_rows, len(embeddings), component_list, args.variance_threshold)
    plot_comparison(results, args.output_dir)


if __name__ == "__main__":
    main()
