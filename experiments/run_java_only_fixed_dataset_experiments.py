"""Run Experiments 1, 2, and 4 on a single fixed Java-only dataset.

This script performs the following steps:
1. Loads the full embedding-based dataset.
2. Filters to Java-only projects (Apache Java repos).
3. Saves a Java-only dataset and a frozen Java-only sample.
4. Runs existing experiment code on that shared sample, with output folders clearly
   distinguished as Java-only experiment artifacts.

The result is a fair, comparable Java-only benchmark for JIT-only, CodeBERT-only,
 and JIT+CodeBERT fusion models.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.sampling import filter_commits_by_date, stratified_sample

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

SOURCE_DATA = PROJECT_ROOT / "data" / "final_multilanguage_dataset_with_embeddings_pca384.csv"
JAVA_ONLY_DATA = PROJECT_ROOT / "data" / "java_only_final_multilanguage_dataset_with_embeddings_pca384.csv"
JAVA_ONLY_SAMPLE = PROJECT_ROOT / "results" / "data" / "java_only_frozen_sample_dataset.csv"

EXP1_OUT = PROJECT_ROOT / "results" / "java_only_experiment_1"
EXP2_OUT = PROJECT_ROOT / "results" / "java_only_experiment_2"
EXP4A_OUT = PROJECT_ROOT / "results" / "java_only_experiment_4a"


def _load_source_dataset() -> pd.DataFrame:
    if not SOURCE_DATA.exists():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE_DATA}")

    df = pd.read_csv(SOURCE_DATA)
    required = {"commit_id", "project", "buggy", "author_date"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Source dataset missing required columns: {missing}")
    return df


def _filter_java_only(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["project"] = df["project"].astype(str)
    java_mask = df["project"].str.startswith("apache/")
    java_df = df[java_mask].copy()
    if "author_date" in java_df.columns:
        java_df["author_date"] = pd.to_datetime(
            java_df["author_date"], unit="s", errors="coerce"
        )
    java_df = java_df.sort_values("author_date", kind="mergesort").reset_index(drop=True)
    logger.info("Java-only dataset rows: %d", len(java_df))
    return java_df


def _create_java_dataset() -> pd.DataFrame:
    src = _load_source_dataset()
    java_df = _filter_java_only(src)
    JAVA_ONLY_DATA.parent.mkdir(parents=True, exist_ok=True)
    java_df.to_csv(JAVA_ONLY_DATA, index=False)
    logger.info("Saved Java-only dataset to %s", JAVA_ONLY_DATA)
    return java_df


def _create_java_frozen_sample(java_df: pd.DataFrame) -> pd.DataFrame:
    filtered = filter_commits_by_date(
        java_df,
        timestamp_column="author_date",
        start_year=2018,
        end_year=2026,
    )
    sample = stratified_sample(
        filtered,
        group_columns=("project", "buggy"),
        target_size=10000,
        random_state=42,
    )
    JAVA_ONLY_SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(JAVA_ONLY_SAMPLE, index=False)
    logger.info("Saved Java-only frozen sample to %s", JAVA_ONLY_SAMPLE)
    logger.info("Java-only sample rows: %d", len(sample))
    return sample


def _run_experiment(script_path: Path, output_dir: Path, data_path: Path, override_kwargs: dict | None = None):
    """Import an existing experiment module and override its data/output paths."""
    spec = importlib.util.spec_from_file_location(f"java_only_{script_path.stem}", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if override_kwargs is None:
        override_kwargs = {}

    if hasattr(module, "DATA_PATH"):
        module.DATA_PATH = data_path
    if hasattr(module, "INPUT_PATH"):
        module.INPUT_PATH = data_path
    if hasattr(module, "OUTPUT_DIR"):
        module.OUTPUT_DIR = output_dir
    if hasattr(module, "EXPERIMENT_DIR"):
        module.EXPERIMENT_DIR = output_dir
    if hasattr(module, "METRICS_DIR"):
        module.METRICS_DIR = output_dir / "metrics"
    if hasattr(module, "PREDICTIONS_DIR"):
        module.PREDICTIONS_DIR = output_dir / "predictions"
    if hasattr(module, "PLOTS_DIR"):
        module.PLOTS_DIR = output_dir / "plots"
    if hasattr(module, "MODELS_DIR"):
        module.MODELS_DIR = output_dir / "models"

    for key, value in override_kwargs.items():
        setattr(module, key, value)

    if hasattr(module, "main"):
        logger.info("Running %s with Java-only dataset: %s", script_path.name, data_path)
        module.main()
        logger.info("Completed %s", script_path.name)
    else:
        raise RuntimeError(f"Module {script_path} does not expose a main() function")


def _patch_exp1_to_use_fixed_java_sample() -> None:
    """Experiment 1 resamples internally; patch it to use the fixed Java sample as-is."""
    exp1_script = PROJECT_ROOT / "experiments" / "experiment_1_jit_only.py"
    spec = importlib.util.spec_from_file_location("exp1_java_only", exp1_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {exp1_script}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.DATA_PATH = JAVA_ONLY_SAMPLE
    module.EXPERIMENT_DIR = EXP1_OUT
    module.METRICS_DIR = EXP1_OUT / "metrics"
    module.PREDICTIONS_DIR = EXP1_OUT / "predictions"
    module.PLOTS_DIR = EXP1_OUT / "plots"
    module.MODELS_DIR = EXP1_OUT / "models"

    def _identity_sample(df, *args, **kwargs):
        return df

    module.create_stratified_subset = _identity_sample
    module.main()


def main() -> None:
    logger.info("Creating Java-only dataset subset")
    java_df = _create_java_dataset()
    logger.info("Creating Java-only frozen sample")
    _create_java_frozen_sample(java_df)

    logger.info("Running Java-only Experiment 1")
    _patch_exp1_to_use_fixed_java_sample()

    logger.info("Running Java-only Experiment 2")
    _run_experiment(
        PROJECT_ROOT / "experiments" / "experiment_2_codebert_only.py",
        EXP2_OUT,
        JAVA_ONLY_SAMPLE,
    )

    logger.info("Running Java-only Experiment 4A")
    _run_experiment(
        PROJECT_ROOT / "experiments" / "exp4" / "experiment_4a_jit_codebert.py",
        EXP4A_OUT,
        JAVA_ONLY_SAMPLE,
    )

    logger.info("Java-only dataset experiments complete.")
    logger.info("Java-only dataset: %s", JAVA_ONLY_DATA)
    logger.info("Java-only sample: %s", JAVA_ONLY_SAMPLE)
    logger.info("Experiment outputs: %s, %s, %s", EXP1_OUT, EXP2_OUT, EXP4A_OUT)


if __name__ == "__main__":
    main()
