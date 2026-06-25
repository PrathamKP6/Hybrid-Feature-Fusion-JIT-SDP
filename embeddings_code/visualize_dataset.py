"""
Embeddings Visualization and Analysis Script

Purpose:
    Analyze and visualize CodeBERT embeddings generated from commit
    messages and code diffs. Produces statistical summaries, dimensionality
    reduction plots, dataset quality checks, and embedding diagnostics.

Input:
    - final_multilanguage_dataset_with_embeddings.csv

Outputs:
    - Dataset overview plots
    - Embedding statistics visualizations
    - PCA projections
    - t-SNE projections
    - UMAP projections
    - Text length distributions
    - Summary report

Use Case:
    Validate embedding quality and explore the structure of the
    feature space before training defect prediction models.
"""

import os
import sys
import argparse
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_embeddings(csv_path, max_rows=None):
    """Load embeddings from CSV file."""
    print(f"Loading embeddings from {csv_path}...")
    df = pd.read_csv(csv_path, nrows=max_rows)
    
    # Parse embedding column
    if 'embedding' in df.columns:
        df['embedding'] = df['embedding'].apply(lambda x: np.array(json.loads(x)) if isinstance(x, str) else x)
    
    embeddings = np.array([e for e in df['embedding']])
    return df, embeddings


def plot_embedding_statistics(embeddings, output_dir="./visualizations"):
    """Plot statistics about the embeddings."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating embedding statistics plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Embedding Statistics (768-D CodeBERT)', fontsize=16, fontweight='bold')
    
    # Mean values per dimension
    mean_values = embeddings.mean(axis=0)
    axes[0, 0].hist(mean_values, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Distribution of Mean Values Across Dimensions')
    axes[0, 0].set_xlabel('Mean Value')
    axes[0, 0].set_ylabel('Frequency')
    
    # Std values per dimension
    std_values = embeddings.std(axis=0)
    axes[0, 1].hist(std_values, bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
    axes[0, 1].set_title('Distribution of Std Dev Across Dimensions')
    axes[0, 1].set_xlabel('Std Dev')
    axes[0, 1].set_ylabel('Frequency')
    
    # Norm distribution (magnitude of each embedding vector)
    norms = np.linalg.norm(embeddings, axis=1)
    axes[1, 0].hist(norms, bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[1, 0].set_title('Distribution of Embedding Norms (L2)')
    axes[1, 0].set_xlabel('Norm')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(norms.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {norms.mean():.3f}')
    axes[1, 0].legend()
    
    # Sparsity (percentage of near-zero values)
    sparsity = np.mean(np.abs(embeddings) < 0.01) * 100
    axes[1, 1].text(0.5, 0.5, f'Sparsity Analysis\n\nNear-zero values (<0.01): {sparsity:.2f}%\nMean absolute value: {np.mean(np.abs(embeddings)):.4f}\nMax value: {embeddings.max():.4f}\nMin value: {embeddings.min():.4f}',
                    ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '01_embedding_statistics.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_tsne_projection(embeddings, df, output_dir="./visualizations", sample_size=5000):
    """Create t-SNE 2D projection of embeddings."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating t-SNE projection (sampling {sample_size} points)...")
    
    try:
        from sklearn.manifold import TSNE
        
        # Sample if too large
        if len(embeddings) > sample_size:
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            embeddings_sample = embeddings[indices]
            df_sample = df.iloc[indices].reset_index(drop=True)
        else:
            embeddings_sample = embeddings
            df_sample = df.reset_index(drop=True)
        
        # Compute t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, verbose=1)
        embeddings_2d = tsne.fit_transform(embeddings_sample)
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 10))
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            alpha=0.6, s=20, c=np.arange(len(embeddings_2d)), 
                            cmap='viridis', edgecolors='none')
        ax.set_title(f't-SNE 2D Projection of CodeBERT Embeddings (n={len(embeddings_sample)})', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        plt.colorbar(scatter, ax=ax, label='Sample Index')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, '02_tsne_projection.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
        
    except ImportError:
        print("⚠ scikit-learn not installed. Skipping t-SNE projection.")


def plot_umap_projection(embeddings, df, output_dir="./visualizations", sample_size=5000):
    """Create UMAP 2D projection of embeddings."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating UMAP projection (sampling {sample_size} points)...")
    
    try:
        import umap
        
        # Sample if too large
        if len(embeddings) > sample_size:
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            embeddings_sample = embeddings[indices]
            df_sample = df.iloc[indices].reset_index(drop=True)
        else:
            embeddings_sample = embeddings
            df_sample = df.reset_index(drop=True)
        
        # Compute UMAP
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1, verbose=1)
        embeddings_2d = reducer.fit_transform(embeddings_sample)
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 10))
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            alpha=0.6, s=20, c=np.arange(len(embeddings_2d)), 
                            cmap='plasma', edgecolors='none')
        ax.set_title(f'UMAP 2D Projection of CodeBERT Embeddings (n={len(embeddings_sample)})', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('UMAP Component 1')
        ax.set_ylabel('UMAP Component 2')
        plt.colorbar(scatter, ax=ax, label='Sample Index')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, '03_umap_projection.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
        
    except ImportError:
        print("⚠ umap-learn not installed. Install with: pip install umap-learn")


def plot_pca_projection(embeddings, df, output_dir="./visualizations"):
    """Create PCA 2D projection of embeddings."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating PCA projection...")
    
    try:
        from sklearn.decomposition import PCA
        
        # Compute PCA
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 10))
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            alpha=0.6, s=20, c=np.arange(len(embeddings_2d)), 
                            cmap='coolwarm', edgecolors='none')
        ax.set_title(f'PCA 2D Projection of CodeBERT Embeddings (Explained Var: {pca.explained_variance_ratio_.sum():.1%})', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.colorbar(scatter, ax=ax, label='Sample Index')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, '04_pca_projection.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
        
    except ImportError:
        print("⚠ scikit-learn not installed. Skipping PCA projection.")


def plot_dataset_overview(df, output_dir="./visualizations"):
    """Plot overview statistics of the dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating dataset overview...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Dataset Overview', fontsize=16, fontweight='bold')
    
    # Dataset size
    axes[0, 0].text(0.5, 0.5, f'Total Rows: {len(df):,}\nTotal Columns: {len(df.columns)}',
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[0, 0].axis('off')
    axes[0, 0].set_title('Dataset Size')
    
    # Columns info
    col_info = f"Columns:\n" + "\n".join([f"• {col}" for col in df.columns[:10]])
    if len(df.columns) > 10:
        col_info += f"\n... and {len(df.columns) - 10} more"
    axes[0, 1].text(0.05, 0.95, col_info, ha='left', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[0, 1].axis('off')
    axes[0, 1].set_title('Columns')
    
    # Data types
    dtype_counts = df.dtypes.value_counts()
    axes[1, 0].barh(range(len(dtype_counts)), dtype_counts.values, color='lightcoral', edgecolor='black')
    axes[1, 0].set_yticks(range(len(dtype_counts)))
    axes[1, 0].set_yticklabels([str(dt) for dt in dtype_counts.index])
    axes[1, 0].set_xlabel('Count')
    axes[1, 0].set_title('Data Types Distribution')
    for i, v in enumerate(dtype_counts.values):
        axes[1, 0].text(v + 0.1, i, str(v), va='center')
    
    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        axes[1, 1].barh(range(len(missing)), missing.values, color='lightgreen', edgecolor='black')
        axes[1, 1].set_yticks(range(len(missing)))
        axes[1, 1].set_yticklabels(missing.index)
        axes[1, 1].set_xlabel('Missing Count')
        axes[1, 1].set_title('Missing Values')
        for i, v in enumerate(missing.values):
            axes[1, 1].text(v + 0.1, i, str(v), va='center')
    else:
        axes[1, 1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        axes[1, 1].axis('off')
    axes[1, 1].set_title('Data Quality')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '00_dataset_overview.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_text_length_distribution(df, output_dir="./visualizations"):
    """Plot distribution of text lengths in dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating text length distribution...")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Text Length Distribution', fontsize=14, fontweight='bold')
    
    text_columns = []
    for col in df.columns:
        if df[col].dtype == 'object' and col != 'embedding':
            text_columns.append(col)
    
    for idx, col in enumerate(text_columns[:3]):
        lengths = df[col].fillna('').astype(str).str.len()
        axes[idx].hist(lengths, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[idx].set_title(f'{col} Length Distribution')
        axes[idx].set_xlabel('Character Count')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_yscale('log')
        axes[idx].text(0.98, 0.97, f'Mean: {lengths.mean():.0f}\nMax: {lengths.max():.0f}',
                      transform=axes[idx].transAxes, ha='right', va='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '05_text_length_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def generate_summary_report(df, embeddings, output_dir="./visualizations"):
    """Generate a text summary report of the dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating summary report...")
    
    # ✅ FIX: safely compute duplicates (ignore embedding column)
    try:
        duplicate_count = df.drop(columns=['embedding'], errors='ignore').duplicated().sum()
    except Exception:
        duplicate_count = "Error computing duplicates"
    
    report = f"""
=============================================================================
                    EMBEDDINGS DATASET SUMMARY REPORT
=============================================================================

Dataset Shape:
  - Total Rows: {len(df):,}
  - Total Columns: {len(df.columns)}
  - Embeddings Dimension: {embeddings.shape[1]}

Columns:
{chr(10).join([f"  - {col}: {df[col].dtype}" for col in df.columns])}

Embedding Statistics:
  - Shape: {embeddings.shape}
  - Min Value: {embeddings.min():.6f}
  - Max Value: {embeddings.max():.6f}
  - Mean Value: {embeddings.mean():.6f}
  - Std Dev: {embeddings.std():.6f}
  - L2 Norm (mean): {np.linalg.norm(embeddings, axis=1).mean():.6f}
  - Sparsity (< 0.01): {np.mean(np.abs(embeddings) < 0.01) * 100:.2f}%

Data Quality:
  - Missing Values: {df.isnull().sum().sum()}
  - Duplicate Rows: {duplicate_count}
  
Text Columns Analysis:
"""
    
    for col in df.columns:
        if df[col].dtype == 'object' and col != 'embedding':
            lengths = df[col].fillna('').astype(str).str.len()
            report += f"\n  {col}:\n"
            report += f"    - Mean Length: {lengths.mean():.0f}\n"
            report += f"    - Max Length: {lengths.max():.0f}\n"
            report += f"    - Min Length: {lengths.min():.0f}\n"
    
    report += "\n=============================================================================\n"
    
    report_path = os.path.join(output_dir, 'SUMMARY_REPORT.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✓ Saved: {report_path}")
    print("\n" + report)


def main():
    parser = argparse.ArgumentParser(description='Visualize CodeBERT embeddings dataset')
    parser.add_argument('--input', type=str, default=str(Path('data') / 'final_multilanguage_dataset_with_embeddings.csv'),
                       help='Path to the embeddings CSV file')
    parser.add_argument('--output-dir', type=str, default=str(Path('embeddings_and_pca_code') / 'visualizations'),
                       help='Output directory for visualizations')
    parser.add_argument('--sample-size', type=int, default=5000,
                       help='Sample size for t-SNE/UMAP (smaller=faster)')
    parser.add_argument('--max-rows', type=int, default=None,
                       help='Maximum rows to load (for testing)')
    parser.add_argument('--skip-tsne', action='store_true', help='Skip t-SNE projection')
    parser.add_argument('--skip-umap', action='store_true', help='Skip UMAP projection')
    parser.add_argument('--skip-pca', action='store_true', help='Skip PCA projection')
    
    args = parser.parse_args()
    
    # Load data
    try:
        df, embeddings = load_embeddings(args.input, args.max_rows)
        print(f"✓ Loaded {len(df)} rows with {embeddings.shape[1]}-D embeddings\n")
    except FileNotFoundError:
        print(f"✗ File not found: {args.input}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        sys.exit(1)
    
    # Create visualizations
    print("=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70 + "\n")
    
    plot_dataset_overview(df, args.output_dir)
    plot_embedding_statistics(embeddings, args.output_dir)
    plot_text_length_distribution(df, args.output_dir)
    
    if not args.skip_pca:
        plot_pca_projection(embeddings, df, args.output_dir)
    
    if not args.skip_tsne:
        plot_tsne_projection(embeddings, df, args.output_dir, args.sample_size)
    
    if not args.skip_umap:
        plot_umap_projection(embeddings, df, args.output_dir, args.sample_size)
    
    generate_summary_report(df, embeddings, args.output_dir)
    
    print("\n" + "=" * 70)
    print(f"✓ All visualizations saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
