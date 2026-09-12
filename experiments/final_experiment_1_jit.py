"""Final full-dataset Experiment 1: JIT metrics only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.final_experiment_pipeline import JIT_FEATURES, PROJECT_ROOT, run_experiment


if __name__ == "__main__":
    run_experiment("final_experiment_1_jit_only", JIT_FEATURES, PROJECT_ROOT / "results" / "final_experiment_1")