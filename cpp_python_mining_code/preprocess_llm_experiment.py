import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
import joblib


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent.parent / "data"

LLM_DIR = BASE / "llm_experiment_dataset"
CHRONO_DIR = BASE / "chronological_split_dataset"

OUTPUT_DIR = BASE / "llm_experiment_preprocessed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FILES
# ============================================================

splits = ["train", "validation", "test"]


# ============================================================
# CATEGORICAL AND NUMERICAL FEATURES
# ============================================================

categorical_features = [
    "intent",
    "change",
    "risk",
    "complexity",
    "scope",
    "test",
    "security"
]

numerical_features = [
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
    "security_margin"
]


# ============================================================
# READ DATASETS AND ADD BUGGY LABEL
# ============================================================

print("=" * 70)
print("LLM EXPERIMENT DATASET PREPROCESSING")
print("=" * 70)

datasets = {}

for split in splits:

    llm_file = LLM_DIR / f"{split}.csv"
    chrono_file = CHRONO_DIR / f"{split}.csv"

    print(f"\nReading {split}.csv...")

    llm_df = pd.read_csv(
        llm_file,
        dtype={"commit_id": str},
        low_memory=False
    )

    chrono_df = pd.read_csv(
        chrono_file,
        dtype={"commit_id": str},
        low_memory=False
    )

    # --------------------------------------------------------
    # Check commit IDs
    # --------------------------------------------------------

    if "commit_id" not in llm_df.columns:
        raise ValueError(
            f"commit_id missing in {llm_file}"
        )

    if "commit_id" not in chrono_df.columns:
        raise ValueError(
            f"commit_id missing in {chrono_file}"
        )

    if "buggy" not in chrono_df.columns:
        raise ValueError(
            f"buggy column missing in {chrono_file}"
        )

    # --------------------------------------------------------
    # Check duplicate IDs in chronological dataset
    # --------------------------------------------------------

    if chrono_df["commit_id"].duplicated().any():

        duplicates = chrono_df.loc[
            chrono_df["commit_id"].duplicated(keep=False),
            "commit_id"
        ].unique()

        raise ValueError(
            f"Duplicate commit IDs found in chronological "
            f"{split} dataset: {len(duplicates)}"
        )

    # --------------------------------------------------------
    # Create commit_id -> buggy mapping
    # --------------------------------------------------------

    label_map = dict(
        zip(
            chrono_df["commit_id"],
            chrono_df["buggy"]
        )
    )

    # --------------------------------------------------------
    # Check missing labels
    # --------------------------------------------------------

    missing_labels = [
        cid
        for cid in llm_df["commit_id"]
        if cid not in label_map
    ]

    if missing_labels:

        print(
            f"WARNING: {len(missing_labels)} commit IDs "
            f"have no buggy label."
        )

        missing_file = (
            OUTPUT_DIR /
            f"{split}_missing_buggy_labels.txt"
        )

        with open(missing_file, "w", encoding="utf-8") as f:
            for cid in missing_labels:
                f.write(str(cid) + "\n")

        raise ValueError(
            f"Missing buggy labels found in {split}. "
            f"See {missing_file}"
        )

    # --------------------------------------------------------
    # Add buggy column
    # --------------------------------------------------------

    llm_df["buggy"] = llm_df["commit_id"].map(label_map)

    # --------------------------------------------------------
    # Check that label count matches
    # --------------------------------------------------------

    if len(llm_df) != len(chrono_df):

        raise ValueError(
            f"Row count mismatch in {split}: "
            f"LLM={len(llm_df)}, "
            f"Chronological={len(chrono_df)}"
        )

    # --------------------------------------------------------
    # Check buggy values
    # --------------------------------------------------------

    print(
        f"{split.capitalize()} rows: {len(llm_df)}"
    )

    print(
        "Buggy distribution:"
    )

    print(
        llm_df["buggy"].value_counts(dropna=False)
    )

    datasets[split] = llm_df


# ============================================================
# CHECK REQUIRED FEATURES
# ============================================================

print("\n" + "=" * 70)
print("CHECKING FEATURES")
print("=" * 70)

required_features = (
    categorical_features +
    numerical_features
)

for split, df in datasets.items():

    missing_features = [
        col
        for col in required_features
        if col not in df.columns
    ]

    if missing_features:

        raise ValueError(
            f"{split} is missing features: "
            f"{missing_features}"
        )

print("OK: All required LLM features are present.")


# ============================================================
# PREPARE X AND y
# ============================================================

X = {}
y = {}

for split, df in datasets.items():

    # Keep only actual predictive features
    X[split] = df[
        categorical_features +
        numerical_features
    ].copy()

    # Target
    y[split] = df["buggy"].copy()


# ============================================================
# NORMALIZE BUGGY TARGET
# ============================================================

def normalize_buggy(value):

    if pd.isna(value):
        return None

    if isinstance(value, bool):
        return int(value)

    value_str = str(value).strip().lower()

    if value_str in ["1", "true"]:
        return 1

    if value_str in ["0", "false"]:
        return 0

    try:

        numeric_value = int(float(value))

        if numeric_value in [0, 1]:
            return numeric_value

    except ValueError:
        pass

    return None


for split in splits:

    y[split] = y[split].apply(normalize_buggy)

    if y[split].isna().any():

        raise ValueError(
            f"Invalid or missing buggy values found in {split}"
        )

    y[split] = y[split].astype(int)


# ============================================================
# ONE-HOT ENCODING
# FIT ONLY ON TRAIN
# ============================================================

print("\n" + "=" * 70)
print("ONE-HOT ENCODING")
print("=" * 70)

print("\nFitting encoder ONLY on training data...")

try:

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

except TypeError:

    # Compatibility with older scikit-learn versions
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False
    )


encoder.fit(
    X["train"][categorical_features]
)


# ============================================================
# TRANSFORM ALL SPLITS
# ============================================================

processed = {}

for split in splits:

    print(f"\nProcessing {split}...")

    # --------------------------------------------------------
    # Encode categorical features
    # --------------------------------------------------------

    encoded_array = encoder.transform(
        X[split][categorical_features]
    )

    encoded_columns = encoder.get_feature_names_out(
        categorical_features
    )

    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoded_columns,
        index=X[split].index
    )

    # --------------------------------------------------------
    # Numerical features remain unchanged
    # --------------------------------------------------------

    numeric_df = X[split][numerical_features].copy()

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    final_X = pd.concat(
        [
            encoded_df,
            numeric_df
        ],
        axis=1
    )

    # --------------------------------------------------------
    # Add target as LAST column
    # --------------------------------------------------------

    final_X["buggy"] = y[split].values

    processed[split] = final_X

    print(
        f"{split.capitalize()} processed shape: "
        f"{final_X.shape}"
    )


# ============================================================
# SAVE PROCESSED DATASETS
# ============================================================

print("\n" + "=" * 70)
print("SAVING PROCESSED DATASETS")
print("=" * 70)

for split in splits:

    output_file = OUTPUT_DIR / f"{split}.csv"

    processed[split].to_csv(
        output_file,
        index=False
    )

    print(
        f"{split}.csv -> {output_file}"
    )


# ============================================================
# SAVE ENCODER
# ============================================================

encoder_file = OUTPUT_DIR / "one_hot_encoder.joblib"

joblib.dump(
    encoder,
    encoder_file
)

print(
    f"\nEncoder saved to:\n{encoder_file}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED")
print("=" * 70)

print("\nFinal datasets:")

for split in splits:

    print(
        f"  {split}.csv : "
        f"{processed[split].shape}"
    )

print("\nOriginal LLM columns removed:")
print("  - commit_id")
print("  - model")
print("  - schema_version")

print("\nCategorical features one-hot encoded:")
for col in categorical_features:
    print(f"  - {col}")

print("\nNumerical features kept unchanged:")
for col in numerical_features:
    print(f"  - {col}")

print("\nTarget:")
print("  - buggy")

print("\nNo scaling was performed.")

print("\nEncoder was fitted ONLY on training data.")

print("=" * 70)