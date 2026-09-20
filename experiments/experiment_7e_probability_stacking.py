"""
EXPERIMENT 7E: PROBABILITY STACKING — THREE-WAY JIT + CODEBERT + LLM FUSION

Purpose:
    Evaluate the complete three-way fusion of:
        1. Traditional JIT metrics
        2. CodeBERT semantic representations
        3. Structured LLM reasoning features

Feature configuration:
    JIT:       12 features
    CodeBERT: 384 PCA features
    LLM:       51 features
    Total:     113 features

Models:
    - Random Forest
    - XGBoost
    - LightGBM

Methodology:
    - Project-wise chronological 70/15/15 split
    - PCA fitted on TRAIN ONLY
    - One-hot encoding fitted on TRAIN ONLY
    - Numerical median imputation fitted on TRAIN ONLY
    - No resampling
    - No random train/test split
    - commit_id used ONLY for alignment/verification
    - Test set used only for final evaluation
    - Classification threshold = 0.5
"""

from pathlib import Path
import ast
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier
import matplotlib.pyplot as plt

try:
    from lightgbm import LGBMClassifier
except ImportError:
    raise ImportError(
        "\nLightGBM is not installed in the current virtual environment.\n"
        "Install it with:\n"
        "    pip install lightgbm\n"
    )

warnings.filterwarnings("ignore")


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CODEBERT_DIR = PROJECT_ROOT / "results" / "pca"
LLM_DIR = PROJECT_ROOT / "data" / "llm_experiment_dataset"

# IMPORTANT:
# Use the canonical chronological split files for JIT.
# These contain the 12 JIT metrics and 768-D original embeddings.
JIT_DIR = PROJECT_ROOT / "results" / "chronological_splits"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "experiment_7e_probability_stacking"
)

MODEL_DIR = OUTPUT_DIR / "models"
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
PLOTS_DIR = OUTPUT_DIR / "plots"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# INPUT FILES
# ============================================================================

CODEBERT_FILES = {
    "train": CODEBERT_DIR / "train_pca384.csv",
    "validation": CODEBERT_DIR / "validation_pca384.csv",
    "test": CODEBERT_DIR / "test_pca384.csv",
}

LLM_FILES = {
    "train": LLM_DIR / "train.csv",
    "validation": LLM_DIR / "validation.csv",
    "test": LLM_DIR / "test.csv",
}

JIT_FILES = {
    "train": JIT_DIR / "train.csv",
    "validation": JIT_DIR / "validation.csv",
    "test": JIT_DIR / "test.csv",
}


# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

JIT_FEATURES = [
    "la",
    "ld",
    "nf",
    "ns",
    "nd",
    "ent",
    "ndev",
    "age",
    "nuc",
    "aexp",
    "arexp",
    "asexp",
]

LLM_CATEGORICAL_COLUMNS = [
    "intent",
    "change",
    "risk",
    "complexity",
    "scope",
    "test",
    "security",
]

LLM_NUMERICAL_COLUMNS = [
    "intent_confidence",
    "change_confidence",
    "risk_confidence",
    "complexity_confidence",
    "scope_confidence",
    "test_confidence",
    "security_confidence",
    "intent_margin",
    "change_margin",
    "risk_margin",
    "complexity_margin",
    "scope_margin",
    "test_margin",
    "security_margin",
]

TARGET_COLUMN = "buggy"
ID_COLUMN = "commit_id"

CODEBERT_SOURCE_DIM = 384
CODEBERT_DIM = 384
LLM_DIM = 51
JIT_DIM = 12
TOTAL_FEATURES = 447

THRESHOLD = 0.5
RANDOM_STATE = 42


# ============================================================================
# HELPERS
# ============================================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_file_exists(path):
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file does not exist:\n{path}"
        )


def normalize_target(series):
    """
    Normalize buggy labels to integer 0/1.
    """

    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")

        if values.isna().any():
            raise ValueError(
                "Target column contains invalid numeric values."
            )

        unique_values = set(values.unique())

        if not unique_values.issubset({0, 1}):
            raise ValueError(
                f"Target must contain only 0/1. "
                f"Found: {sorted(unique_values)}"
            )

        return values.astype(int)

    mapping = {
        "0": 0,
        "1": 1,
        "false": 0,
        "true": 1,
        "non-buggy": 0,
        "nonbuggy": 0,
        "buggy": 1,
        "bug": 1,
        "clean": 0,
    }

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    if normalized.isna().any():
        bad_values = series[normalized.isna()].unique()

        raise ValueError(
            f"Unknown target values: {bad_values}"
        )

    return normalized.astype(int)


def verify_ids(name, df):

    if ID_COLUMN not in df.columns:
        raise ValueError(
            f"{name}: '{ID_COLUMN}' column is missing."
        )

    duplicate_count = df[ID_COLUMN].duplicated().sum()

    print(
        f"{name}: rows={len(df):,}, "
        f"duplicate commit IDs={duplicate_count:,}"
    )

    if duplicate_count != 0:
        raise ValueError(
            f"{name} contains duplicate commit IDs."
        )


def verify_no_overlap(train_df, val_df, test_df):

    train_ids = set(train_df[ID_COLUMN])
    val_ids = set(val_df[ID_COLUMN])
    test_ids = set(test_df[ID_COLUMN])

    train_val = len(train_ids & val_ids)
    train_test = len(train_ids & test_ids)
    val_test = len(val_ids & test_ids)

    print(f"Train / Validation: {train_val}")
    print(f"Train / Test:       {train_test}")
    print(f"Validation / Test:  {val_test}")

    if train_val or train_test or val_test:
        raise ValueError(
            "Cross-split commit overlap detected."
        )

    print("\n[OK] No cross-split commit overlap.")


# ============================================================================
# EMBEDDING PARSER
# ============================================================================

def parse_embedding_column(series, expected_dim=384):

    vectors = []

    for idx, value in enumerate(series):

        if isinstance(value, np.ndarray):
            vector = value.tolist()

        elif isinstance(value, list):
            vector = value

        elif isinstance(value, str):

            text = value.strip()

            try:
                vector = json.loads(text)
            except Exception:

                try:
                    vector = ast.literal_eval(text)
                except Exception as exc:

                    raise ValueError(
                        f"Could not parse embedding at row "
                        f"{idx}: {exc}"
                    )

        else:

            raise ValueError(
                f"Unsupported embedding type at row "
                f"{idx}: {type(value)}"
            )

        if not isinstance(vector, (list, tuple)):

            raise ValueError(
                f"Embedding at row {idx} is not a list."
            )

        if len(vector) != expected_dim:

            raise ValueError(
                f"Embedding dimension mismatch at row {idx}: "
                f"expected {expected_dim}, got {len(vector)}"
            )

        vectors.append(vector)

    matrix = np.asarray(
        vectors,
        dtype=np.float32,
    )

    if matrix.shape[1] != expected_dim:

        raise ValueError(
            f"Embedding matrix has shape {matrix.shape}; "
            f"expected second dimension {expected_dim}."
        )

    return matrix


# ============================================================================
# GENERIC DATASET ALIGNMENT
# ============================================================================

def align_by_commit_id(reference_df, other_df, name):

    verify_ids(
        f"Reference {name}",
        reference_df,
    )

    verify_ids(
        f"Other {name}",
        other_df,
    )

    reference_ids = set(
        reference_df[ID_COLUMN]
    )

    other_ids = set(
        other_df[ID_COLUMN]
    )

    missing_in_other = (
        reference_ids - other_ids
    )

    missing_in_reference = (
        other_ids - reference_ids
    )

    if missing_in_other:

        raise ValueError(
            f"{name}: {len(missing_in_other):,} "
            f"reference IDs missing in other dataset."
        )

    if missing_in_reference:

        raise ValueError(
            f"{name}: {len(missing_in_reference):,} "
            f"other IDs missing in reference dataset."
        )

    other_indexed = other_df.set_index(
        ID_COLUMN
    )

    aligned = other_indexed.loc[
        reference_df[ID_COLUMN]
    ].reset_index()

    if not np.array_equal(
        reference_df[ID_COLUMN].values,
        aligned[ID_COLUMN].values,
    ):

        raise ValueError(
            f"{name}: commit_id alignment failed."
        )

    print(
        f"[OK] {name}: "
        f"{len(reference_df):,} rows aligned by commit_id."
    )

    return aligned


# ============================================================================
# LLM PREPROCESSING
# ============================================================================

def build_llm_features(
    train_llm,
    validation_llm,
    test_llm,
):

    print_header(
        "ONE-HOT ENCODER"
    )

    print(
        "Encoder will be FIT ON TRAINING DATA ONLY."
    )

    required = (
        [ID_COLUMN, TARGET_COLUMN]
        + LLM_CATEGORICAL_COLUMNS
        + LLM_NUMERICAL_COLUMNS
    )

    for name, df in [
        ("train", train_llm),
        ("validation", validation_llm),
        ("test", test_llm),
    ]:

        missing = [
            col
            for col in required
            if col not in df.columns
        ]

        if missing:

            raise ValueError(
                f"{name}: missing columns: {missing}"
            )

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float32,
    )

    X_train_cat = encoder.fit_transform(
        train_llm[
            LLM_CATEGORICAL_COLUMNS
        ].astype(str)
    )

    X_validation_cat = encoder.transform(
        validation_llm[
            LLM_CATEGORICAL_COLUMNS
        ].astype(str)
    )

    X_test_cat = encoder.transform(
        test_llm[
            LLM_CATEGORICAL_COLUMNS
        ].astype(str)
    )

    categorical_names = (
        encoder
        .get_feature_names_out(
            LLM_CATEGORICAL_COLUMNS
        )
        .tolist()
    )

    print(
        f"Generated categorical features: "
        f"{len(categorical_names)}"
    )

    if len(categorical_names) != 37:

        raise ValueError(
            "Expected 37 one-hot features, "
            f"got {len(categorical_names)}."
        )

    print(
        "[OK] 37 categorical one-hot features."
    )

    print(
        "\nCalculating numerical feature medians "
        "ONLY from training data."
    )

    train_num = (
        train_llm[
            LLM_NUMERICAL_COLUMNS
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    validation_num = (
        validation_llm[
            LLM_NUMERICAL_COLUMNS
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    test_num = (
        test_llm[
            LLM_NUMERICAL_COLUMNS
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_num = imputer.fit_transform(
        train_num
    )

    X_validation_num = imputer.transform(
        validation_num
    )

    X_test_num = imputer.transform(
        test_num
    )

    X_train_num = X_train_num.astype(
        np.float32
    )

    X_validation_num = (
        X_validation_num.astype(np.float32)
    )

    X_test_num = X_test_num.astype(
        np.float32
    )

    X_train_llm = np.hstack([
        X_train_cat,
        X_train_num,
    ])

    X_validation_llm = np.hstack([
        X_validation_cat,
        X_validation_num,
    ])

    X_test_llm = np.hstack([
        X_test_cat,
        X_test_num,
    ])

    feature_names = (
        categorical_names
        + LLM_NUMERICAL_COLUMNS
    )

    if len(feature_names) != 51:

        raise ValueError(
            f"Expected 51 LLM features, "
            f"got {len(feature_names)}."
        )

    print(
        "\n[OK] "
        "37 categorical + 14 numerical = "
        "51 LLM features."
    )

    return (
        X_train_llm,
        X_validation_llm,
        X_test_llm,
        feature_names,
        encoder,
        imputer,
    )


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(
    model,
    X,
    y,
    split_name,
):

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    accuracy = accuracy_score(
        y,
        predictions,
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    mcc = matthews_corrcoef(
        y,
        predictions,
    )

    roc_auc = roc_auc_score(
        y,
        probabilities,
    )

    pr_auc = average_precision_score(
        y,
        probabilities,
    )

    cm = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    )

    print_header(
        f"{split_name.upper()} RESULTS"
    )

    print(
        f"ACCURACY    : {accuracy:.4f}"
    )

    print(
        f"PRECISION   : {precision:.4f}"
    )

    print(
        f"RECALL      : {recall:.4f}"
    )

    print(
        f"F1          : {f1:.4f}"
    )

    print(
        f"MCC         : {mcc:.4f}"
    )

    print(
        f"ROC_AUC     : {roc_auc:.4f}"
    )

    print(
        f"PR_AUC      : {pr_auc:.4f}"
    )

    print("\nConfusion Matrix:")
    print(cm)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "probabilities": probabilities,
        "predictions": predictions,
    }


def clean_metrics(metrics):

    return {
        key: value
        for key, value in metrics.items()
        if key not in [
            "probabilities",
            "predictions",
        ]
    }


def plot_roc_pr(y_true, probabilities, model_name, split_name):
    """
    Generate ROC and Precision-Recall curves using the same probabilities
    used for the reported ROC-AUC and PR-AUC metrics.

    PR-AUC is reported using average_precision_score, so the plot annotation
    also uses Average Precision for consistency with the experiment metrics.
    """
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    roc_auc = roc_auc_score(y_true, probabilities)

    plt.figure(figsize=(7, 6))
    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{model_name} (ROC-AUC = {roc_auc:.4f})",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        label="Random classifier",
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        f"{model_name} - {split_name.capitalize()} ROC Curve"
    )
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    roc_path = (
        PLOTS_DIR
        / f"{model_name.lower().replace(' ', '_')}_"
          f"{split_name.lower()}_roc.png"
    )
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(
        y_true,
        probabilities,
    )
    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    plt.figure(figsize=(7, 6))
    plt.plot(
        recall,
        precision,
        linewidth=2,
        label=f"{model_name} (PR-AUC = {pr_auc:.4f})",
    )
    baseline = np.mean(y_true)
    plt.axhline(
        baseline,
        linestyle="--",
        linewidth=1,
        label=f"Positive prevalence = {baseline:.4f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        f"{model_name} - {split_name.capitalize()} "
        "Precision-Recall Curve"
    )
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    pr_path = (
        PLOTS_DIR
        / f"{model_name.lower().replace(' ', '_')}_"
          f"{split_name.lower()}_pr.png"
    )
    plt.savefig(pr_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"{split_name.capitalize()} ROC plot: {roc_path}")
    print(f"{split_name.capitalize()} PR plot:  {pr_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print_header("EXPERIMENT 7E: PROBABILITY STACKING — THREE-WAY JIT + CODEBERT + LLM FUSION")
    print("""
Methodology:
  Split: Project-wise chronological 70/15/15
  Base feature space: 12 JIT + 384 CodeBERT + 51 LLM = 447
  Base models: Random Forest, XGBoost, LightGBM
  Stacking: validation probabilities -> Logistic Regression meta-model
  Meta-model training: VALIDATION ONLY
  Test set: final evaluation only
  PCA: 384 components, fitted TRAIN ONLY
  One-hot encoding: TRAIN ONLY
  Median imputation: TRAIN ONLY
  Resampling: NOT USED
  Random train/test split: NOT USED
  commit_id: alignment/verification only
  Base classification threshold: 0.5
  Stacking threshold: 0.5
""")

    print_header("CHECKING INPUT FILES")
    for path in CODEBERT_FILES.values(): check_file_exists(path)
    for path in LLM_FILES.values(): check_file_exists(path)
    for path in JIT_FILES.values(): check_file_exists(path)
    print("[OK] All required files exist.")

    codebert, llm, jit = {}, {}, {}
    for split in ["train", "validation", "test"]:
        print_header(f"LOADING {split.upper()} DATA")
        codebert[split] = pd.read_csv(CODEBERT_FILES[split]); verify_ids(f"CodeBERT {split}", codebert[split])
        llm[split] = pd.read_csv(LLM_FILES[split]); verify_ids(f"LLM {split}", llm[split])
        jit[split] = pd.read_csv(JIT_FILES[split]); verify_ids(f"JIT {split}", jit[split])
        print(f"CodeBERT: {len(codebert[split]):,} rows, {len(codebert[split].columns)} columns")
        print(f"LLM:      {len(llm[split]):,} rows, {len(llm[split].columns)} columns")
        print(f"JIT:      {len(jit[split]):,} rows, {len(jit[split].columns)} columns")

    print_header("SPLIT VERIFICATION")
    expected_sizes={"train":41998,"validation":8998,"test":9000}
    for split,expected in expected_sizes.items():
        sizes=[len(codebert[split]),len(llm[split]),len(jit[split])]
        print(f"{split.capitalize():12s}: CodeBERT={sizes[0]:,}, LLM={sizes[1]:,}, JIT={sizes[2]:,}")
        if any(x!=expected for x in sizes): raise ValueError(f"{split}: unexpected row count.")

    print_header("COMMIT-ID ALIGNMENT")
    aligned_llm={}; aligned_jit={}
    for split in ["train","validation","test"]:
        aligned_llm[split]=align_by_commit_id(codebert[split],llm[split],f"LLM {split}")
        aligned_jit[split]=align_by_commit_id(codebert[split],jit[split],f"JIT {split}")

    print_header("CROSS-SPLIT COMMIT-ID VERIFICATION")
    verify_no_overlap(codebert["train"],codebert["validation"],codebert["test"])

    print_header("TARGET ALIGNMENT VERIFICATION")
    y={}
    for split in ["train","validation","test"]:
        yc=normalize_target(codebert[split][TARGET_COLUMN]).to_numpy()
        yl=normalize_target(aligned_llm[split][TARGET_COLUMN]).to_numpy()
        yj=normalize_target(aligned_jit[split][TARGET_COLUMN]).to_numpy()
        if not (np.array_equal(yc,yl) and np.array_equal(yc,yj)):
            raise ValueError(f"{split}: target labels do not match across all three datasets.")
        y[split]=yc
        print(f"[OK] {split.capitalize()}: JIT, CodeBERT and LLM labels match.")

    print_header("PREPARING CODEBERT FEATURES")
    X_codebert={}
    for split in ["train","validation","test"]:
        X_codebert[split]=parse_embedding_column(codebert[split]["embedding"],CODEBERT_SOURCE_DIM)
        print(f"X_{split} CodeBERT: {X_codebert[split].shape}")
        if X_codebert[split].shape[1] != 384: raise ValueError(f"{split}: expected 384 CodeBERT components.")
    print("[OK] CodeBERT source representation verified: 384 dimensions.")

    print_header("PREPARING JIT FEATURES")
    X_jit={}
    for split in ["train","validation","test"]:
        missing=[c for c in JIT_FEATURES if c not in aligned_jit[split].columns]
        if missing: raise ValueError(f"JIT {split}: missing features: {missing}")
        X_jit[split]=aligned_jit[split][JIT_FEATURES].apply(pd.to_numeric,errors="coerce").to_numpy(dtype=np.float32)
        print(f"X_{split} JIT: {X_jit[split].shape}")

    print_header("PREPARING LLM REASONING FEATURES")
    X_train_llm,X_validation_llm,X_test_llm,llm_feature_names,encoder,imputer=build_llm_features(aligned_llm["train"],aligned_llm["validation"],aligned_llm["test"])
    X_llm={"train":X_train_llm,"validation":X_validation_llm,"test":X_test_llm}
    print(f"LLM features: {len(llm_feature_names)}")
    if len(llm_feature_names)!=51: raise ValueError(f"Expected 51 LLM features, got {len(llm_feature_names)}")

    print_header("THREE-WAY FEATURE FUSION")
    X={}
    for split in ["train","validation","test"]:
        X[split]=np.hstack([X_jit[split],X_codebert[split],X_llm[split]])
        print(f"X_{split}: {X[split].shape}")
    jit_feature_names=[f"jit_{f}" for f in JIT_FEATURES]
    codebert_feature_names=[f"codebert_pca_{i+1}" for i in range(384)]
    feature_names=jit_feature_names+codebert_feature_names+llm_feature_names
    if len(feature_names)!=447: raise ValueError(f"Expected 447 features, got {len(feature_names)}")
    for split in X:
        if X[split].shape[1]!=447: raise ValueError(f"{split}: expected 447 features, got {X[split].shape[1]}")
    print("[OK] 12 JIT + 384 CodeBERT + 51 LLM = 447 features.")

    print_header("FEATURE QUALITY CHECK")
    for split in X:
        nan_count=np.isnan(X[split]).sum(); inf_count=np.isinf(X[split]).sum()
        print(f"{split.capitalize():12s}: NaN={nan_count:,}, Inf={inf_count:,}")
        if nan_count or inf_count: raise ValueError(f"{split}: NaN or infinite values detected.")
    print("[OK] No NaN or infinite values remain.")

    print_header("TARGET DISTRIBUTION")
    for split in y:
        pos=int(y[split].sum()); total=len(y[split]); neg=total-pos
        print(f"{split.capitalize():12s}: buggy={pos:,} / {total:,} ({pos/total:.2%}), non-buggy={neg:,}")

    train_positive=int(y["train"].sum()); train_negative=len(y["train"])-train_positive
    scale_pos_weight=train_negative/train_positive
    print_header("CLASS IMBALANCE")
    print(f"Training negatives: {train_negative:,}")
    print(f"Training positives: {train_positive:,}")
    print(f"XGBoost scale_pos_weight: {scale_pos_weight:.4f}")
    print(f"LightGBM scale_pos_weight: {scale_pos_weight:.4f}")

    print_header("TRAINING BASE MODELS")
    models={
      "random_forest": RandomForestClassifier(n_estimators=300,max_features="sqrt",class_weight="balanced",random_state=RANDOM_STATE,n_jobs=-1),
      "xgboost": XGBClassifier(n_estimators=300,max_depth=6,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,eval_metric="logloss",scale_pos_weight=scale_pos_weight,random_state=RANDOM_STATE,n_jobs=-1),
      "lightgbm": LGBMClassifier(n_estimators=300,max_depth=6,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,scale_pos_weight=scale_pos_weight,objective="binary",random_state=RANDOM_STATE,n_jobs=-1,verbosity=-1),
    }
    base_validation={}; base_test={}; base_metrics={}
    for name,model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X["train"],y["train"])
        base_validation[name]=model.predict_proba(X["validation"])[:,1]
        base_test[name]=model.predict_proba(X["test"])[:,1]
        base_metrics[name]={
          "validation": clean_metrics(evaluate_model(model,X["validation"],y["validation"],"validation")),
          "test": clean_metrics(evaluate_model(model,X["test"],y["test"],"test")),
        }
        joblib.dump(model,MODEL_DIR/f"{name}_base.joblib")
        pd.DataFrame({ID_COLUMN:codebert["test"][ID_COLUMN].values,"actual_buggy":y["test"],"predicted_probability":base_test[name],"predicted_buggy":(base_test[name]>=0.5).astype(int)}).to_csv(PREDICTIONS_DIR/f"{name}_test_predictions.csv",index=False)

    print_header("STACKING META-FEATURES")
    meta_feature_names=["rf_probability","xgb_probability","lgbm_probability"]
    Z_validation=np.column_stack([base_validation["random_forest"],base_validation["xgboost"],base_validation["lightgbm"]])
    Z_test=np.column_stack([base_test["random_forest"],base_test["xgboost"],base_test["lightgbm"]])
    print(f"Validation meta-features: {Z_validation.shape}")
    print(f"Test meta-features:       {Z_test.shape}")
    if Z_validation.shape!=(8998,3) or Z_test.shape!=(9000,3): raise ValueError("Unexpected stacking meta-feature shape.")
    print("[OK] Meta-features contain validation/test probabilities from the three base models.")

    from sklearn.linear_model import LogisticRegression
    meta_model=LogisticRegression(class_weight="balanced",random_state=RANDOM_STATE,max_iter=1000)
    print_header("TRAINING LOGISTIC REGRESSION META-MODEL")
    print("Meta-model is fitted on VALIDATION probabilities only.")
    meta_model.fit(Z_validation,y["validation"])
    stacking_validation_prob=meta_model.predict_proba(Z_validation)[:,1]
    stacking_test_prob=meta_model.predict_proba(Z_test)[:,1]
    stacking_validation_pred=(stacking_validation_prob>=THRESHOLD).astype(int)
    stacking_test_pred=(stacking_test_prob>=THRESHOLD).astype(int)

    def stacking_metrics(y_true,prob,pred):
        return {
          "accuracy":accuracy_score(y_true,pred),"precision":precision_score(y_true,pred,zero_division=0),"recall":recall_score(y_true,pred,zero_division=0),"f1":f1_score(y_true,pred,zero_division=0),"mcc":matthews_corrcoef(y_true,pred),"roc_auc":roc_auc_score(y_true,prob),"pr_auc":average_precision_score(y_true,prob),"confusion_matrix":confusion_matrix(y_true,pred).tolist(),"probabilities":prob,"predictions":pred,
        }
    stack_val=stacking_metrics(y["validation"],stacking_validation_prob,stacking_validation_pred)
    stack_test=stacking_metrics(y["test"],stacking_test_prob,stacking_test_pred)

    print_header("STACKING VALIDATION RESULTS")
    for k in ["accuracy","precision","recall","f1","mcc","roc_auc","pr_auc"]: print(f"{k.upper():12s}: {stack_val[k]:.4f}")
    print("Confusion Matrix:"); print(np.array(stack_val["confusion_matrix"]))
    print_header("STACKING TEST RESULTS")
    for k in ["accuracy","precision","recall","f1","mcc","roc_auc","pr_auc"]: print(f"{k.upper():12s}: {stack_test[k]:.4f}")
    print("Confusion Matrix:"); print(np.array(stack_test["confusion_matrix"]))

    joblib.dump(meta_model,MODEL_DIR/"logistic_regression_meta_model.joblib")
    pd.DataFrame({ID_COLUMN:codebert["test"][ID_COLUMN].values,"actual_buggy":y["test"],"predicted_probability":stacking_test_prob,"predicted_buggy":stacking_test_pred}).to_csv(PREDICTIONS_DIR/"stacking_test_predictions.csv",index=False)
    pd.DataFrame(Z_validation,columns=meta_feature_names).to_csv(PREDICTIONS_DIR/"stacking_validation_meta_features.csv",index=False)
    pd.DataFrame(Z_test,columns=meta_feature_names).to_csv(PREDICTIONS_DIR/"stacking_test_meta_features.csv",index=False)

    print_header("GENERATING STACKING ROC / PR PLOTS")
    plot_roc_pr(y["validation"],stacking_validation_prob,"Probability Stacking","validation")
    plot_roc_pr(y["test"],stacking_test_prob,"Probability Stacking","test")

    results_df=pd.DataFrame([{
      "model":"probability_stacking",
      "validation_accuracy":stack_val["accuracy"],"validation_precision":stack_val["precision"],"validation_recall":stack_val["recall"],"validation_f1":stack_val["f1"],"validation_mcc":stack_val["mcc"],"validation_roc_auc":stack_val["roc_auc"],"validation_pr_auc":stack_val["pr_auc"],
      "test_accuracy":stack_test["accuracy"],"test_precision":stack_test["precision"],"test_recall":stack_test["recall"],"test_f1":stack_test["f1"],"test_mcc":stack_test["mcc"],"test_roc_auc":stack_test["roc_auc"],"test_pr_auc":stack_test["pr_auc"]
    }])
    results_path=METRICS_DIR/"experiment_7e_results.csv"; results_df.to_csv(results_path,index=False)
    detailed={"base_models":base_metrics,"stacking":{"validation":clean_metrics({k:v for k,v in stack_val.items() if k not in ["probabilities","predictions"]}),"test":clean_metrics({k:v for k,v in stack_test.items() if k not in ["probabilities","predictions"]})}}
    with open(METRICS_DIR/"experiment_7e_metrics.json","w",encoding="utf-8") as f: json.dump(detailed,f,indent=2)
    with open(OUTPUT_DIR/"stacking_meta_features.json","w",encoding="utf-8") as f: json.dump({"features":meta_feature_names,"meta_model":"LogisticRegression","meta_training_split":"validation_only","threshold":THRESHOLD},f,indent=2)
    with open(OUTPUT_DIR/"experiment_7e_configuration.json","w",encoding="utf-8") as f:
        json.dump({"experiment":"7E","name":"Probability Stacking — RF + XGBoost + LightGBM on 447-feature three-way fusion","feature_counts":{"jit":12,"codebert":384,"llm":51,"total":447},"pca":"384 components, fitted train only","one_hot":"train only","median_imputation":"train only","base_models":["Random Forest","XGBoost","LightGBM"],"meta_model":"Logistic Regression","meta_training":"validation probabilities only","threshold":THRESHOLD,"resampling":False,"random_split":False,"test_used_for_tuning":False},f,indent=2)

    print_header("EXPERIMENT 7E COMPLETED")
    print("""Feature configuration:
  JIT features:       12
  CodeBERT features: 384
  LLM features:       51
  Total base features: 447
  Stacking meta-features: 3 probabilities

Base models:
  Random Forest
  XGBoost
  LightGBM

Meta-model:
  Logistic Regression
  Trained on validation probabilities only

Dataset:
  Train:       41,998
  Validation:  8,998
  Test:        9,000
  Total:       59,996
""")
    print("Final STACKING TEST performance:")
    print(results_df.to_string(index=False,float_format=lambda x:f"{x:.6f}"))
    print(f"\nResults saved to:\n  {results_path}")
    print(f"\nModels saved to:\n  {MODEL_DIR}")
    print(f"\nPlots saved to:\n  {PLOTS_DIR}")

if __name__ == "__main__":
    main()
