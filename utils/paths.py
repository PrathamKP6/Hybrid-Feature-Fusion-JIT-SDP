from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
EMBEDDINGS_DIR = DATA_DIR

FINAL_DATASET_CSV = DATA_DIR / "final_multilanguage_dataset.csv"
FINAL_DATASET_EMBEDDINGS_CSV = DATA_DIR / "final_multilanguage_dataset_with_embeddings.csv"
FINAL_DATASET_EMBEDDINGS_PCA384_CSV = DATA_DIR / "final_multilanguage_dataset_with_embeddings_pca384.csv"
FINAL_DATASET_RAW_CSV = DATA_DIR / "final_dataset.csv"
FROZEN_SAMPLE_CSV = RESULTS_DIR / "frozen_sample_dataset.csv"
FROZEN_SAMPLE_COMMIT_IDS_CSV = RESULTS_DIR / "frozen_sample_commit_ids.csv"

EXPERIMENT1_DIR = RESULTS_DIR / "experiment_1"
EXPERIMENT2_DIR = RESULTS_DIR / "experiment_2"
