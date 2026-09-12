"""Final full-dataset Experiment 2: CodeBERT PCA384 only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.final_experiment_pipeline import PCA_FEATURES, PROJECT_ROOT, run_experiment


if __name__ == "__main__":
    run_experiment("final_experiment_2_codebert_pca384_only", PCA_FEATURES, PROJECT_ROOT / "results" / "final_experiment_2")