"""Final full-dataset Experiment 4: JIT plus CodeBERT PCA384."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.final_experiment_pipeline import JIT_FEATURES, PCA_FEATURES, PROJECT_ROOT, run_experiment


if __name__ == "__main__":
    run_experiment("final_experiment_4_jit_plus_codebert_pca384", JIT_FEATURES + PCA_FEATURES, PROJECT_ROOT / "results" / "final_experiment_4")