"""
Experiment 5 — JIT + LLM Feature Preparation
==============================================

This version starts from the AUTHORITATIVE RAW LLM DATASET because the
previous one-hot encoded LLM files removed commit_id.

Pipeline
--------
1. Load canonical chronological JIT splits.
2. Load raw LLM splits from data/llm_experiment_dataset/.
3. Verify commit_id alignment and split integrity.
4. Fit OneHotEncoder on TRAIN categorical LLM features ONLY.
5. Transform train/validation/test with that same encoder.
6. Keep the 14 LLM confidence/margin features unchanged.
7. Preserve commit_id in the processed LLM data.
8. Merge processed LLM features with canonical JIT features by commit_id,
   separately within train/validation/test.
9. Validate the final datasets.
10. Save the merged JIT + LLM datasets and preprocessing metadata.

No:
- random splitting
- PCA
- resampling
- fitting preprocessing on validation/test
- row-wise/positional merging
- source CSV modification

Expected rows:
    train       41,998
    validation   8,998
    test         9,000
    total       59,996

Expected LLM feature count:
    37 one-hot categorical features + 14 numerical confidence/margin
    features = 51 LLM features.

JIT features:
    la, ld, nf, nd, ns, ent, ndev, age, nuc, aexp, arexp, asexp

If Experiment 1 used a different exact JIT feature list, change
JIT_FEATURES below to match Experiment 1 exactly.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CANONICAL_DIR = PROJECT_ROOT / "results" / "chronological_splits"
LLM_DIR = PROJECT_ROOT / "data" / "llm_experiment_dataset"

OUTPUT_DIR = PROJECT_ROOT / "results" / "experiment_5_jit_llm"
DATASET_DIR = OUTPUT_DIR / "datasets"
LLM_PREPROCESSED_DIR = OUTPUT_DIR / "llm_preprocessed"
METADATA_DIR = OUTPUT_DIR / "metadata"

CANONICAL_PATHS = {
    "train": CANONICAL_DIR / "train.csv",
    "validation": CANONICAL_DIR / "validation.csv",
    "test": CANONICAL_DIR / "test.csv",
}

LLM_PATHS = {
    "train": LLM_DIR / "train.csv",
    "validation": LLM_DIR / "validation.csv",
    "test": LLM_DIR / "test.csv",
}

OUTPUT_PATHS = {
    "train": DATASET_DIR / "train_jit_llm.csv",
    "validation": DATASET_DIR / "validation_jit_llm.csv",
    "test": DATASET_DIR / "test_jit_llm.csv",
}

SPLITS = ["train", "validation", "test"]

EXPECTED_ROWS = {
    "train": 41_998,
    "validation": 8_998,
    "test": 9_000,
}

EXPECTED_TOTAL = 59_996

ID_COLUMN = "commit_id"
TARGET = "buggy"


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

CATEGORICAL_FEATURES = [
    "intent",
    "change",
    "risk",
    "complexity",
    "scope",
    "test",
    "security",
]

NUMERICAL_LLM_FEATURES = [
    "intent_confidence",
    "intent_margin",
    "change_confidence",
    "change_margin",
    "risk_confidence",
    "risk_margin",
    "complexity_confidence",
    "complexity_margin",
    "scope_confidence",
    "scope_margin",
    "test_confidence",
    "test_margin",
    "security_confidence",
    "security_margin",
]

JIT_FEATURES = [
    "la",
    "ld",
    "nf",
    "nd",
    "ns",
    "ent",
    "ndev",
    "age",
    "nuc",
    "aexp",
    "arexp",
    "asexp",
]

LLM_RAW_REQUIRED = (
    [ID_COLUMN]
    + CATEGORICAL_FEATURES
    + NUMERICAL_LLM_FEATURES
)

LLM_FEATURE_COUNT_EXPECTED = 51
JIT_FEATURE_COUNT_EXPECTED = len(JIT_FEATURES)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# GENERAL VALIDATION
# ============================================================

def ensure_paths_exist() -> None:
    paths = list(CANONICAL_PATHS.values()) + list(LLM_PATHS.values())

    missing = [str(path) for path in paths if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Required files are missing:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )


def load_csv(path: Path, name: str) -> pd.DataFrame:
    logger.info("Loading %s: %s", name, path)

    df = pd.read_csv(
        path,
        dtype={ID_COLUMN: str},
        low_memory=False,
    )

    logger.info(
        "%s: %s rows × %s columns",
        name,
        f"{len(df):,}",
        len(df.columns),
    )

    if ID_COLUMN not in df.columns:
        raise ValueError(
            f"{name} does not contain '{ID_COLUMN}'."
        )

    if df[ID_COLUMN].isna().any():
        raise ValueError(
            f"{name} contains missing commit_id values."
        )

    if df[ID_COLUMN].duplicated().any():
        count = int(df[ID_COLUMN].duplicated().sum())
        raise ValueError(
            f"{name} contains {count:,} duplicate commit IDs."
        )

    return df


def validate_row_count(
    df: pd.DataFrame,
    split: str,
    source: str,
) -> None:
    expected = EXPECTED_ROWS[split]

    if len(df) != expected:
        raise ValueError(
            f"{source} {split}: expected {expected:,} rows, "
            f"found {len(df):,}."
        )


def normalize_buggy(value):
    """Normalize mixed 0/1/False/True target representations."""
    if pd.isna(value):
        return np.nan

    if isinstance(value, (bool, np.bool_)):
        return int(value)

    if isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        if value in (0, 1):
            return int(value)

    value_str = str(value).strip().lower()

    if value_str in {"0", "false"}:
        return 0

    if value_str in {"1", "true"}:
        return 1

    try:
        numeric_value = int(float(value_str))
        if numeric_value in (0, 1):
            return numeric_value
    except (ValueError, TypeError):
        pass

    raise ValueError(
        f"Unexpected buggy value: {value!r}"
    )


def normalize_canonical_targets(
    canonical: dict[str, pd.DataFrame],
) -> None:
    for split, df in canonical.items():
        if TARGET not in df.columns:
            raise ValueError(
                f"Canonical {split} is missing '{TARGET}'."
            )

        df[TARGET] = df[TARGET].apply(normalize_buggy)

        if df[TARGET].isna().any():
            raise ValueError(
                f"Canonical {split} contains missing/invalid buggy labels."
            )

        df[TARGET] = df[TARGET].astype(np.int8)

        unique = set(df[TARGET].unique())

        if not unique.issubset({0, 1}):
            raise ValueError(
                f"Canonical {split} has invalid target values: {unique}"
            )


def verify_cross_split_ids(
    datasets: dict[str, pd.DataFrame],
    name: str,
) -> None:
    id_sets = {
        split: set(df[ID_COLUMN])
        for split, df in datasets.items()
    }

    train_val = id_sets["train"] & id_sets["validation"]
    train_test = id_sets["train"] & id_sets["test"]
    val_test = id_sets["validation"] & id_sets["test"]

    logger.info(
        "%s cross-split overlaps: train/val=%d, train/test=%d, val/test=%d",
        name,
        len(train_val),
        len(train_test),
        len(val_test),
    )

    if train_val or train_test or val_test:
        raise ValueError(
            f"{name}: cross-split commit-ID overlap detected."
        )


def verify_exact_split_alignment(
    canonical: dict[str, pd.DataFrame],
    llm: dict[str, pd.DataFrame],
) -> None:
    print("\n" + "=" * 80)
    print("COMMIT-ID ALIGNMENT")
    print("=" * 80)

    for split in SPLITS:
        canonical_ids = set(canonical[split][ID_COLUMN])
        llm_ids = set(llm[split][ID_COLUMN])

        missing_from_llm = canonical_ids - llm_ids
        extra_in_llm = llm_ids - canonical_ids

        matched = canonical_ids & llm_ids

        print(f"\n{split.upper()}")
        print(f"  Canonical IDs : {len(canonical_ids):,}")
        print(f"  LLM IDs       : {len(llm_ids):,}")
        print(f"  Matched IDs   : {len(matched):,}")
        print(f"  Missing in LLM: {len(missing_from_llm):,}")
        print(f"  Extra in LLM  : {len(extra_in_llm):,}")

        if missing_from_llm:
            sample = list(missing_from_llm)[:10]
            raise ValueError(
                f"{split}: canonical IDs missing from LLM. "
                f"Sample: {sample}"
            )

        if extra_in_llm:
            sample = list(extra_in_llm)[:10]
            raise ValueError(
                f"{split}: LLM IDs absent from canonical data. "
                f"Sample: {sample}"
            )


# ============================================================
# LLM PREPROCESSING
# ============================================================

def validate_llm_raw_schema(
    llm: dict[str, pd.DataFrame],
) -> None:
    for split, df in llm.items():
        missing = [
            column
            for column in LLM_RAW_REQUIRED
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"LLM {split} is missing required columns: {missing}"
            )


def fit_encoder(
    train_llm: pd.DataFrame,
) -> OneHotEncoder:
    print("\n" + "=" * 80)
    print("FITTING ONE-HOT ENCODER")
    print("=" * 80)

    print(
        "\nFitting categorical encoder ONLY on training data."
    )

    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )

    encoder.fit(
        train_llm[CATEGORICAL_FEATURES]
    )

    encoded_columns = encoder.get_feature_names_out(
        CATEGORICAL_FEATURES
    )

    print(
        f"Categorical input columns : "
        f"{len(CATEGORICAL_FEATURES)}"
    )
    print(
        f"One-hot output columns    : "
        f"{len(encoded_columns)}"
    )

    for column in encoded_columns:
        print(f"  - {column}")

    return encoder


def preprocess_llm_split(
    df: pd.DataFrame,
    encoder: OneHotEncoder,
    split: str,
) -> pd.DataFrame:
    print(f"\nProcessing LLM {split}...")

    # Preserve commit_id explicitly.
    output = df[[ID_COLUMN]].copy()

    # Transform categorical features using the encoder fitted on TRAIN.
    encoded_array = encoder.transform(
        df[CATEGORICAL_FEATURES]
    )

    encoded_columns = encoder.get_feature_names_out(
        CATEGORICAL_FEATURES
    )

    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoded_columns,
        index=df.index,
    )

    # Confidence/margin values are not fitted/transformed.
    numeric_df = df[
        NUMERICAL_LLM_FEATURES
    ].copy()

    # Force numeric representation and validate.
    for column in NUMERICAL_LLM_FEATURES:
        numeric_df[column] = pd.to_numeric(
            numeric_df[column],
            errors="coerce",
        )

    if numeric_df.isna().any().any():
        bad_columns = [
            c for c in numeric_df.columns
            if numeric_df[c].isna().any()
        ]

        raise ValueError(
            f"LLM {split} has missing/non-numeric values in: "
            f"{bad_columns}"
        )

    final = pd.concat(
        [
            output,
            encoded_df,
            numeric_df,
        ],
        axis=1,
    )

    feature_columns = list(encoded_columns) + NUMERICAL_LLM_FEATURES

    if len(feature_columns) != LLM_FEATURE_COUNT_EXPECTED:
        raise ValueError(
            f"Expected {LLM_FEATURE_COUNT_EXPECTED} LLM features, "
            f"got {len(feature_columns)}."
        )

    values = final[feature_columns].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(values).all():
        raise ValueError(
            f"LLM {split} contains NaN/Inf after preprocessing."
        )

    print(
        f"  Output shape: {final.shape[0]:,} × {final.shape[1]}"
    )
    print(
        f"  Feature count: {len(feature_columns)}"
    )

    return final


# ============================================================
# JIT + LLM MERGE
# ============================================================

def merge_jit_llm(
    canonical_df: pd.DataFrame,
    llm_processed_df: pd.DataFrame,
    llm_features: list[str],
    split: str,
) -> pd.DataFrame:
    """
    Merge one chronological split.

    Canonical data supplies:
        commit_id, buggy, JIT metrics, metadata

    Processed LLM data supplies:
        commit_id + 51 LLM features

    Merge is explicitly one-to-one on commit_id.
    """

    canonical_ids = set(
        canonical_df[ID_COLUMN]
    )
    llm_ids = set(
        llm_processed_df[ID_COLUMN]
    )

    if canonical_ids != llm_ids:
        raise ValueError(
            f"{split}: canonical and processed LLM ID sets differ."
        )

    merged = canonical_df.merge(
        llm_processed_df[
            [ID_COLUMN] + llm_features
        ],
        on=ID_COLUMN,
        how="inner",
        sort=False,
        validate="one_to_one",
    )

    if len(merged) != len(canonical_df):
        raise ValueError(
            f"{split}: merge changed row count from "
            f"{len(canonical_df):,} to {len(merged):,}."
        )

    if not merged[ID_COLUMN].equals(
        canonical_df[ID_COLUMN]
    ):
        raise ValueError(
            f"{split}: merge changed commit-ID order."
        )

    return merged


def validate_final_dataset(
    df: pd.DataFrame,
    split: str,
    llm_features: list[str],
) -> None:
    model_features = JIT_FEATURES + llm_features

    required = (
        [ID_COLUMN, TARGET]
        + model_features
    )

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Final {split} dataset missing columns: {missing}"
        )

    if df[ID_COLUMN].duplicated().any():
        raise ValueError(
            f"Final {split} contains duplicate commit IDs."
        )

    if df[TARGET].isna().any():
        raise ValueError(
            f"Final {split} contains missing buggy labels."
        )

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError(
            f"Final {split} contains invalid buggy labels."
        )

    values = df[model_features].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(values).all():
        raise ValueError(
            f"Final {split} contains NaN/Inf model features."
        )


def reorder_final_columns(
    df: pd.DataFrame,
    llm_features: list[str],
) -> pd.DataFrame:
    """
    Keep a clear, reproducible column order.

    Model block:
        commit_id
        buggy
        JIT features
        LLM features

    Traceability metadata then follows.
    """

    model_block = [
        ID_COLUMN,
        TARGET,
        *JIT_FEATURES,
        *llm_features,
    ]

    remaining = [
        c
        for c in df.columns
        if c not in model_block
    ]

    return df[
        model_block + remaining
    ].copy()


# ============================================================
# METADATA
# ============================================================

def save_metadata(
    encoder: OneHotEncoder,
    llm_features: list[str],
    final: dict[str, pd.DataFrame],
) -> None:
    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LLM_PREPROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    encoder_path = (
        LLM_PREPROCESSED_DIR
        / "one_hot_encoder.joblib"
    )

    joblib.dump(
        encoder,
        encoder_path,
    )

    feature_schema = {
        "experiment": "Experiment 5 - JIT + LLM",
        "merge_key": ID_COLUMN,
        "target": TARGET,
        "jit_features": JIT_FEATURES,
        "jit_feature_count": len(JIT_FEATURES),
        "llm_categorical_features": CATEGORICAL_FEATURES,
        "llm_numerical_features": NUMERICAL_LLM_FEATURES,
        "llm_feature_count": len(llm_features),
        "total_model_features": (
            len(JIT_FEATURES) + len(llm_features)
        ),
        "llm_feature_order": llm_features,
        "model_feature_order": (
            JIT_FEATURES + llm_features
        ),
        "categorical_encoding": "OneHotEncoder",
        "handle_unknown": "ignore",
        "encoder_fitted_on": "train split only",
        "numerical_llm_features": "kept unchanged",
        "scaling": False,
        "pca": False,
        "resampling": False,
    }

    with open(
        METADATA_DIR / "feature_schema.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            feature_schema,
            f,
            indent=4,
        )

    preparation_metadata = {
        "experiment": "Experiment 5 - JIT + LLM",
        "sources": {
            "canonical": str(CANONICAL_DIR),
            "raw_llm": str(LLM_DIR),
        },
        "outputs": {
            "datasets": str(DATASET_DIR),
            "llm_preprocessed": str(LLM_PREPROCESSED_DIR),
            "metadata": str(METADATA_DIR),
        },
        "merge": {
            "key": ID_COLUMN,
            "type": "inner",
            "validate": "one_to_one",
            "performed_separately_per_split": True,
        },
        "split_policy": (
            "Existing chronological train/validation/test splits "
            "preserved; no new split created."
        ),
        "rows": {
            split: len(df)
            for split, df in final.items()
        },
        "total_rows": sum(
            len(df)
            for df in final.values()
        ),
        "preprocessing": {
            "one_hot_encoder_fitted_on": "train only",
            "one_hot_handle_unknown": "ignore",
            "llm_numeric_features_unchanged": True,
            "pca_applied": False,
            "resampling_applied": False,
            "random_split_applied": False,
        },
        "source_csvs_modified": False,
        "encoder_path": str(encoder_path),
    }

    with open(
        METADATA_DIR / "preparation_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            preparation_metadata,
            f,
            indent=4,
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("\n" + "=" * 80)
    print("EXPERIMENT 5 — JIT + LLM DATASET PREPARATION")
    print("=" * 80)

    print("\nPipeline:")
    print("  Raw LLM data")
    print("       ↓")
    print("  One-hot encoder FIT ON TRAIN ONLY")
    print("       ↓")
    print("  51 LLM features")
    print("       ↓")
    print("  Merge with canonical JIT by commit_id")
    print("       ↓")
    print("  Existing chronological Train / Validation / Test")
    print("=" * 80)

    ensure_paths_exist()

    # --------------------------------------------------------
    # 1. LOAD CANONICAL + RAW LLM
    # --------------------------------------------------------

    canonical = {
        split: load_csv(
            CANONICAL_PATHS[split],
            f"canonical {split}",
        )
        for split in SPLITS
    }

    llm = {
        split: load_csv(
            LLM_PATHS[split],
            f"raw LLM {split}",
        )
        for split in SPLITS
    }

    # --------------------------------------------------------
    # 2. ROW COUNTS
    # --------------------------------------------------------

    for split in SPLITS:
        validate_row_count(
            canonical[split],
            split,
            "canonical",
        )

        validate_row_count(
            llm[split],
            split,
            "raw LLM",
        )

    # --------------------------------------------------------
    # 3. TARGET NORMALIZATION
    # --------------------------------------------------------

    normalize_canonical_targets(canonical)

    # --------------------------------------------------------
    # 4. CROSS-SPLIT ID CHECKS
    # --------------------------------------------------------

    verify_cross_split_ids(
        canonical,
        "Canonical",
    )

    verify_cross_split_ids(
        llm,
        "Raw LLM",
    )

    # --------------------------------------------------------
    # 5. EXACT ID ALIGNMENT
    # --------------------------------------------------------

    verify_exact_split_alignment(
        canonical,
        llm,
    )

    # --------------------------------------------------------
    # 6. SCHEMA VALIDATION
    # --------------------------------------------------------

    validate_llm_raw_schema(llm)

    for split, df in canonical.items():
        missing_jit = [
            feature
            for feature in JIT_FEATURES
            if feature not in df.columns
        ]

        if missing_jit:
            raise ValueError(
                f"Canonical {split} missing JIT features: "
                f"{missing_jit}"
            )

    # --------------------------------------------------------
    # 7. FIT OHE ONLY ON TRAIN
    # --------------------------------------------------------

    encoder = fit_encoder(
        llm["train"]
    )

    # --------------------------------------------------------
    # 8. PREPROCESS LLM SPLITS
    # --------------------------------------------------------

    processed_llm = {}

    for split in SPLITS:
        processed_llm[split] = preprocess_llm_split(
            llm[split],
            encoder,
            split,
        )

    # --------------------------------------------------------
    # 9. VERIFY PROCESSED LLM IDS
    # --------------------------------------------------------

    verify_cross_split_ids(
        processed_llm,
        "Processed LLM",
    )

    verify_exact_split_alignment(
        canonical,
        processed_llm,
    )

    llm_features = (
        list(
            encoder.get_feature_names_out(
                CATEGORICAL_FEATURES
            )
        )
        + NUMERICAL_LLM_FEATURES
    )

    if len(llm_features) != LLM_FEATURE_COUNT_EXPECTED:
        raise ValueError(
            f"Expected 51 LLM features, got {len(llm_features)}."
        )

    # --------------------------------------------------------
    # 10. SAVE PROCESSED LLM DATA WITH commit_id
    # --------------------------------------------------------

    LLM_PREPROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 80)
    print("SAVING ID-PRESERVING LLM PREPROCESSED DATA")
    print("=" * 80)

    for split in SPLITS:
        path = (
            LLM_PREPROCESSED_DIR
            / f"{split}.csv"
        )

        processed_llm[split].to_csv(
            path,
            index=False,
        )

        print(f"  {split}: {path}")

    # --------------------------------------------------------
    # 11. MERGE JIT + LLM
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("MERGING JIT + LLM")
    print("=" * 80)

    final = {}

    for split in SPLITS:
        merged = merge_jit_llm(
            canonical[split],
            processed_llm[split],
            llm_features,
            split,
        )

        merged = reorder_final_columns(
            merged,
            llm_features,
        )

        validate_final_dataset(
            merged,
            split,
            llm_features,
        )

        final[split] = merged

        print(
            f"  {split}: "
            f"{len(merged):,} rows × "
            f"{len(merged.columns)} columns"
        )

    # --------------------------------------------------------
    # 12. FINAL CROSS-SPLIT VALIDATION
    # --------------------------------------------------------

    verify_cross_split_ids(
        final,
        "Final JIT + LLM",
    )

    total_rows = sum(
        len(final[split])
        for split in SPLITS
    )

    if total_rows != EXPECTED_TOTAL:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL:,} total rows; "
            f"found {total_rows:,}."
        )

    for split in SPLITS:
        if len(final[split]) != EXPECTED_ROWS[split]:
            raise ValueError(
                f"Final {split} row count changed."
            )

    # --------------------------------------------------------
    # 13. SAVE FINAL DATASETS
    # --------------------------------------------------------

    DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 80)
    print("SAVING FINAL JIT + LLM DATASETS")
    print("=" * 80)

    for split in SPLITS:
        final[split].to_csv(
            OUTPUT_PATHS[split],
            index=False,
        )

        print(
            f"  {split}: {OUTPUT_PATHS[split]}"
        )

    # --------------------------------------------------------
    # 14. SAVE METADATA + ENCODER
    # --------------------------------------------------------

    save_metadata(
        encoder,
        llm_features,
        final,
    )

    # --------------------------------------------------------
    # 15. FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("EXPERIMENT 5 PREPARATION COMPLETE")
    print("=" * 80)

    print("\nRows:")
    print(f"  Train       : {len(final['train']):,}")
    print(f"  Validation  : {len(final['validation']):,}")
    print(f"  Test        : {len(final['test']):,}")
    print(f"  Total       : {total_rows:,}")

    print("\nModel feature dimensions:")
    print(f"  JIT         : {len(JIT_FEATURES)}")
    print(f"  LLM         : {len(llm_features)}")
    print(
        f"  JIT + LLM   : "
        f"{len(JIT_FEATURES) + len(llm_features)}"
    )

    print("\nLeakage controls:")
    print("  [OK] Existing chronological splits preserved")
    print("  [OK] No new random split")
    print("  [OK] One-hot encoder fitted on TRAIN only")
    print("  [OK] Validation transformed using train encoder")
    print("  [OK] Test transformed using train encoder")
    print("  [OK] Merge performed on commit_id")
    print("  [OK] One-to-one merge enforced")
    print("  [OK] No cross-split commit overlap")
    print("  [OK] No PCA")
    print("  [OK] No resampling")
    print("  [OK] buggy excluded from model feature block")
    print("  [OK] Source datasets not modified")

    print("\nFinal feature block:")
    print("  JIT features:")
    for feature in JIT_FEATURES:
        print(f"    - {feature}")

    print("\n  LLM features:")
    for feature in llm_features:
        print(f"    - {feature}")

    print("=" * 80)


if __name__ == "__main__":
    main()
