
"""
EXPERIMENT: LLM REDUNDANCY-REDUCED FEATURE FUSION
==================================================

Purpose
-------
Compare three LLM representations while keeping the same:
    JIT      = 12 features
    CodeBERT = 384 train-only PCA features
    LLM      = one of:
        A. all 51 features
        B. 14 confidence + margin features only
        C. 6 categorical semantic dimensions + 14 confidence/margin

IMPORTANT:
The PCA CSV files in this repository store the 384-D PCA vector inside
the single `embedding` column as a JSON/list string. They do NOT contain
pca_1 ... pca_384 columns.

Methodology
-----------
- Existing project-wise chronological 70/15/5? No: 70/15/15 split.
- Existing PCA was fit on TRAIN only.
- LLM one-hot encoder fit on TRAIN only.
- Numerical median imputation fit on TRAIN only.
- No resampling.
- No random train/validation/test split.
- Test is evaluated after the representations are fixed.
- RF, XGBoost, LightGBM.
"""

from pathlib import Path
import ast
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PCA_DIR = PROJECT_ROOT / "results" / "pca"
CANONICAL_DIR = PROJECT_ROOT / "results" / "chronological_splits"
LLM_DIR = PROJECT_ROOT / "data" / "llm_experiment_dataset"

OUT_DIR = PROJECT_ROOT / "results" / "experiment_llm_reduced"
PLOTS_DIR = OUT_DIR / "plots"

OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# FEATURES
# ============================================================

JIT_FEATURES = [
    "la", "ld", "nf", "ns", "nd",
    "ent", "ndev", "age", "nuc",
    "aexp", "arexp", "asexp"
]

LLM_CATEGORICAL = [
    "intent",
    "change",
    "risk",
    "complexity",
    "scope",
    "test",
    "security",
]

LLM_NUMERICAL = [
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

TARGET = "buggy"
ID_COL = "commit_id"
EXPECTED_PCA_DIM = 384

# ============================================================
# TARGET
# ============================================================

def normalize_target(s):
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)

    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)

    mapping = {
        "1": 1,
        "0": 0,
        "true": 1,
        "false": 0,
        "buggy": 1,
        "non-buggy": 0,
        "non_buggy": 0,
        "yes": 1,
        "no": 0,
    }

    out = (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    numeric = pd.to_numeric(s, errors="coerce")
    out = out.where(out.notna(), numeric)

    return out.fillna(0).astype(int)


# ============================================================
# PCA EMBEDDING PARSER
# ============================================================

def parse_embedding(value):
    """
    Parse one 384-D PCA vector from the embedding column.

    Supports:
      - JSON list strings
      - Python list strings
      - actual lists/arrays
    """
    if isinstance(value, np.ndarray):
        arr = value.astype(np.float32)

    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value, dtype=np.float32)

    elif pd.isna(value):
        raise ValueError("Missing embedding value")

    else:
        text = str(value).strip()

        try:
            arr = np.asarray(json.loads(text), dtype=np.float32)
        except Exception:
            try:
                arr = np.asarray(ast.literal_eval(text), dtype=np.float32)
            except Exception as e:
                raise ValueError(
                    f"Could not parse embedding value: {text[:100]}"
                ) from e

    arr = arr.reshape(-1)

    if len(arr) != EXPECTED_PCA_DIM:
        raise ValueError(
            f"Expected PCA dimension {EXPECTED_PCA_DIM}, "
            f"got {len(arr)}"
        )

    return arr


def load_pca_matrix(df, split):
    if "embedding" not in df.columns:
        raise ValueError(
            f"`embedding` column not found in {split} PCA file. "
            f"Available columns: {list(df.columns)}"
        )

    print(f"  Parsing 384-D PCA embeddings for {split}...")

    matrix = np.vstack(
        [parse_embedding(v) for v in df["embedding"].values]
    ).astype(np.float32)

    if matrix.shape != (len(df), EXPECTED_PCA_DIM):
        raise ValueError(
            f"Unexpected PCA matrix shape for {split}: "
            f"{matrix.shape}"
        )

    if not np.isfinite(matrix).all():
        raise ValueError(f"NaN/Inf found in PCA embeddings for {split}")

    print(f"  PCA matrix: {matrix.shape}")

    return matrix


# ============================================================
# OHE
# ============================================================

def make_ohe():
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
            dtype=np.float32,
        )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(
            y, predictions, zero_division=0
        ),
        "recall": recall_score(
            y, predictions, zero_division=0
        ),
        "f1": f1_score(
            y, predictions, zero_division=0
        ),
        "mcc": matthews_corrcoef(y, predictions),
        "roc_auc": roc_auc_score(y, probabilities),
        "pr_auc": average_precision_score(y, probabilities),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def print_metrics(label, m):
    print(
        f"{label}: "
        f"Acc={m['accuracy']:.4f} "
        f"Prec={m['precision']:.4f} "
        f"Recall={m['recall']:.4f} "
        f"F1={m['f1']:.4f} "
        f"MCC={m['mcc']:.4f} "
        f"ROC-AUC={m['roc_auc']:.4f} "
        f"PR-AUC={m['pr_auc']:.4f}"
    )

    print(
        f"  CM=[[{m['tn']}, {m['fp']}], "
        f"[{m['fn']}, {m['tp']}]]"
    )


# ============================================================
# PLOTS
# ============================================================

def plot_roc_pr(
    y,
    probabilities,
    model_name,
    split_name,
    variant_name,
):
    fpr, tpr, _ = roc_curve(y, probabilities)

    plt.figure(figsize=(7, 6))
    plt.plot(
        fpr,
        tpr,
        label=f"ROC-AUC={roc_auc_score(y, probabilities):.4f}",
    )
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        f"{variant_name} - {model_name} - {split_name} ROC"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR
        / f"{variant_name}_{model_name}_{split_name}_roc.png",
        dpi=200,
    )
    plt.close()

    precision, recall, _ = precision_recall_curve(
        y,
        probabilities,
    )

    plt.figure(figsize=(7, 6))
    plt.plot(
        recall,
        precision,
        label=(
            f"PR-AUC="
            f"{average_precision_score(y, probabilities):.4f}"
        ),
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        f"{variant_name} - {model_name} - {split_name} PR"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR
        / f"{variant_name}_{model_name}_{split_name}_pr.png",
        dpi=200,
    )
    plt.close()


# ============================================================
# LOAD AND ALIGN ONE SPLIT
# ============================================================

def load_split(split):
    """
    Load and align three sources:

      1. Canonical chronological split:
         - authoritative target
         - JIT features

      2. PCA split:
         - 384-D CodeBERT PCA vector stored in `embedding`

      3. LLM split:
         - 51 raw LLM features

    The canonical chronological split is used for JIT because the
    PCA CSV in this repository does not necessarily retain the 12
    JIT columns.
    """

    print(f"\nLoading {split}...")

    canonical_path = CANONICAL_DIR / f"{split}.csv"
    pca_path = PCA_DIR / f"{split}_pca384.csv"
    llm_path = LLM_DIR / f"{split}.csv"

    for path in [canonical_path, pca_path, llm_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    canonical = pd.read_csv(canonical_path)
    pca = pd.read_csv(pca_path)
    llm = pd.read_csv(llm_path)

    print(f"  Canonical rows: {len(canonical):,}")
    print(f"  PCA rows:       {len(pca):,}")
    print(f"  LLM rows:       {len(llm):,}")

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_canonical = [ID_COL, TARGET] + JIT_FEATURES

    missing_canonical = [
        c for c in required_canonical
        if c not in canonical.columns
    ]

    if missing_canonical:
        raise ValueError(
            f"Missing canonical columns in {split}: "
            f"{missing_canonical}"
        )

    if ID_COL not in pca.columns:
        raise ValueError(
            f"{ID_COL} missing from PCA file: {pca_path}"
        )

    if ID_COL not in llm.columns:
        raise ValueError(
            f"{ID_COL} missing from LLM file: {llm_path}"
        )

    if "embedding" not in pca.columns:
        raise ValueError(
            f"`embedding` missing from PCA file: {pca_path}"
        )

    # --------------------------------------------------------
    # Duplicate checks
    # --------------------------------------------------------

    for name, df in [
        ("canonical", canonical),
        ("PCA", pca),
        ("LLM", llm),
    ]:
        if df[ID_COL].duplicated().any():
            raise ValueError(
                f"Duplicate commit IDs in {name} {split}"
            )

    # --------------------------------------------------------
    # Row-count checks
    # --------------------------------------------------------

    if len(canonical) != len(pca):
        raise ValueError(
            f"Canonical/PCA row mismatch in {split}: "
            f"{len(canonical)} vs {len(pca)}"
        )

    if len(canonical) != len(llm):
        raise ValueError(
            f"Canonical/LLM row mismatch in {split}: "
            f"{len(canonical)} vs {len(llm)}"
        )

    # --------------------------------------------------------
    # Verify exact ID sets
    # --------------------------------------------------------

    canonical_ids = set(canonical[ID_COL])
    pca_ids = set(pca[ID_COL])
    llm_ids = set(llm[ID_COL])

    if canonical_ids != pca_ids:
        missing = canonical_ids - pca_ids
        extra = pca_ids - canonical_ids
        raise ValueError(
            f"Canonical/PCA ID mismatch in {split}. "
            f"Missing={len(missing)}, Extra={len(extra)}"
        )

    if canonical_ids != llm_ids:
        missing = canonical_ids - llm_ids
        extra = llm_ids - canonical_ids
        raise ValueError(
            f"Canonical/LLM ID mismatch in {split}. "
            f"Missing={len(missing)}, Extra={len(extra)}"
        )

    # --------------------------------------------------------
    # Canonical target
    # --------------------------------------------------------

    canonical[TARGET] = normalize_target(
        canonical[TARGET]
    )

    # --------------------------------------------------------
    # Verify LLM buggy labels if present.
    # Canonical target remains authoritative.
    # --------------------------------------------------------

    if TARGET in llm.columns:

        label_check = canonical[
            [ID_COL, TARGET]
        ].merge(
            llm[[ID_COL, TARGET]],
            on=ID_COL,
            how="inner",
            validate="one_to_one",
            suffixes=("_canonical", "_llm"),
        )

        canonical_y = normalize_target(
            label_check[f"{TARGET}_canonical"]
        )

        llm_y = normalize_target(
            label_check[f"{TARGET}_llm"]
        )

        if not np.array_equal(
            canonical_y.to_numpy(),
            llm_y.to_numpy(),
        ):
            raise ValueError(
                f"LLM buggy labels disagree with canonical "
                f"labels in {split}"
            )

        print("  LLM/canonical buggy labels: MATCH")

        llm = llm.drop(columns=[TARGET])

    # --------------------------------------------------------
    # Verify required LLM columns
    # --------------------------------------------------------

    required_llm = (
        LLM_CATEGORICAL + LLM_NUMERICAL
    )

    missing_llm = [
        c for c in required_llm
        if c not in llm.columns
    ]

    if missing_llm:
        raise ValueError(
            f"Missing LLM columns in {split}: "
            f"{missing_llm}"
        )

    # --------------------------------------------------------
    # Parse 384-D PCA embedding.
    # --------------------------------------------------------

    pca_matrix = load_pca_matrix(
        pca,
        split,
    )

    # --------------------------------------------------------
    # Build final dataframe from CANONICAL JIT + LLM.
    # This is the key fix: JIT comes from chronological_splits.
    # --------------------------------------------------------

    base = canonical[
        [ID_COL, TARGET] + JIT_FEATURES
    ].copy()

    merged = base.merge(
        llm[
            [ID_COL] + required_llm
        ],
        on=ID_COL,
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(canonical):
        raise ValueError(
            f"Canonical/LLM alignment changed row count "
            f"for {split}: "
            f"{len(canonical)} -> {len(merged)}"
        )

    merged[TARGET] = normalize_target(
        merged[TARGET]
    )

    print(
        f"  Final aligned rows: {len(merged):,}"
    )
    print(
        f"  Buggy: {merged[TARGET].sum():,} "
        f"({merged[TARGET].mean():.4f})"
    )
    print(
        f"  JIT features available: {len(JIT_FEATURES)}"
    )

    return merged, pca_matrix


# ============================================================
# FIT / TRANSFORM LLM REPRESENTATION
# ============================================================

def fit_llm_preprocessor(
    train_df,
    categorical_columns,
):
    if categorical_columns:
        ohe = make_ohe()
        ohe.fit(
            train_df[categorical_columns].astype(str)
        )
    else:
        ohe = None

    imputer = SimpleImputer(strategy="median")

    train_numeric = (
        train_df[LLM_NUMERICAL]
        .apply(pd.to_numeric, errors="coerce")
    )

    imputer.fit(train_numeric)

    return ohe, imputer


def transform_llm(
    df,
    ohe,
    imputer,
    categorical_columns,
):
    parts = []
    names = []

    if categorical_columns:
        cat = ohe.transform(
            df[categorical_columns].astype(str)
        ).astype(np.float32)

        parts.append(cat)
        names.extend(
            ohe.get_feature_names_out(
                categorical_columns
            ).tolist()
        )

    numeric = (
        df[LLM_NUMERICAL]
        .apply(pd.to_numeric, errors="coerce")
    )

    numeric = imputer.transform(
        numeric
    ).astype(np.float32)

    parts.append(numeric)
    names.extend(LLM_NUMERICAL)

    X = np.hstack(parts).astype(np.float32)

    return X, names


# ============================================================
# MODELS
# ============================================================

def build_models(scale_pos_weight):
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),

        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
        ),

        "lightgbm": LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            objective="binary",
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        ),
    }


# ============================================================
# RUN ONE VARIANT
# ============================================================

def run_variant(
    variant_name,
    train_df,
    val_df,
    test_df,
    train_pca,
    val_pca,
    test_pca,
    categorical_columns,
):
    print("\n" + "=" * 78)
    print(f"VARIANT: {variant_name}")
    print("=" * 78)

    y_train = train_df[TARGET].to_numpy()
    y_val = val_df[TARGET].to_numpy()
    y_test = test_df[TARGET].to_numpy()

    # --------------------------------------------------------
    # LLM preprocessing: FIT ON TRAIN ONLY
    # --------------------------------------------------------

    ohe, imputer = fit_llm_preprocessor(
        train_df,
        categorical_columns,
    )

    Xllm_train, llm_names = transform_llm(
        train_df,
        ohe,
        imputer,
        categorical_columns,
    )

    Xllm_val, _ = transform_llm(
        val_df,
        ohe,
        imputer,
        categorical_columns,
    )

    Xllm_test, _ = transform_llm(
        test_df,
        ohe,
        imputer,
        categorical_columns,
    )

    # --------------------------------------------------------
    # JIT
    # --------------------------------------------------------

    Xjit_train = (
        train_df[JIT_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )

    Xjit_val = (
        val_df[JIT_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )

    Xjit_test = (
        test_df[JIT_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )

    # --------------------------------------------------------
    # FUSION
    # --------------------------------------------------------

    X_train = np.hstack([
        Xjit_train,
        train_pca,
        Xllm_train,
    ]).astype(np.float32)

    X_val = np.hstack([
        Xjit_val,
        val_pca,
        Xllm_val,
    ]).astype(np.float32)

    X_test = np.hstack([
        Xjit_test,
        test_pca,
        Xllm_test,
    ]).astype(np.float32)

    print(f"JIT features:      {len(JIT_FEATURES)}")
    print(f"CodeBERT features: {EXPECTED_PCA_DIM}")
    print(f"LLM features:      {len(llm_names)}")
    print(f"TOTAL FEATURES:    {X_train.shape[1]}")
    print(
        f"Shapes: train={X_train.shape}, "
        f"val={X_val.shape}, "
        f"test={X_test.shape}"
    )

    if not np.isfinite(X_train).all():
        raise ValueError("NaN/Inf in X_train")
    if not np.isfinite(X_val).all():
        raise ValueError("NaN/Inf in X_val")
    if not np.isfinite(X_test).all():
        raise ValueError("NaN/Inf in X_test")

    # --------------------------------------------------------
    # MODEL CONFIG
    # --------------------------------------------------------

    scale_pos_weight = (
        (len(y_train) - y_train.sum())
        / y_train.sum()
    )

    print(
        f"scale_pos_weight: {scale_pos_weight:.4f}"
    )

    models = build_models(scale_pos_weight)

    rows = []
    plot_data = []

    # Train all models and collect probabilities first.
    # No matplotlib work occurs while models are training.
    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)

        p_val = model.predict_proba(X_val)[:, 1]
        p_test = model.predict_proba(X_test)[:, 1]

        val_metrics = calculate_metrics(
            y_val,
            p_val,
        )

        test_metrics = calculate_metrics(
            y_test,
            p_test,
        )

        print_metrics(
            "VALIDATION",
            val_metrics,
        )

        print_metrics(
            "TEST",
            test_metrics,
        )

        rows.append({
            "variant": variant_name,
            "model": model_name,
            "split": "validation",
            **val_metrics,
        })

        rows.append({
            "variant": variant_name,
            "model": model_name,
            "split": "test",
            **test_metrics,
        })

        plot_data.append(
            (
                model_name,
                y_val.copy(),
                p_val.copy(),
                "validation",
            )
        )

        plot_data.append(
            (
                model_name,
                y_test.copy(),
                p_test.copy(),
                "test",
            )
        )

        # Explicitly release the model before the next model.
        del model

    # --------------------------------------------------------
    # PLOTS AFTER TRAINING
    # --------------------------------------------------------

    print("\nGenerating ROC/PR plots...")

    for (
        model_name,
        y_plot,
        p_plot,
        split_name,
    ) in plot_data:
        plot_roc_pr(
            y_plot,
            p_plot,
            model_name,
            split_name,
            variant_name,
        )

    result = pd.DataFrame(rows)

    result.to_csv(
        OUT_DIR / f"{variant_name}_metrics.csv",
        index=False,
    )

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("LLM REDUNDANCY-REDUCED FEATURE FUSION")
    print("=" * 78)
    print()
    print("JIT: 12")
    print("CodeBERT: 384 PCA dimensions")
    print("JIT source: results/chronological_splits/*.csv")
    print()
    print("LLM representations:")
    print("  A = all 51")
    print("  B = 14 confidence + margin")
    print("  C = 6 categorical dimensions + 14 confidence/margin")
    print()
    print(
        "PCA source: results/pca/*_pca384.csv"
    )
    print(
        "PCA storage: 384-D vector in `embedding` column"
    )
    print()
    print(
        "Split: existing project-wise chronological 70/15/15"
    )
    print("PCA: already fit on TRAIN only")
    print("LLM OHE: fit on TRAIN only")
    print("LLM imputation: fit on TRAIN only")
    print("No resampling")
    print("No random split")
    print("Matplotlib backend: Agg (non-GUI)")
    print("Test evaluated only after training")
    print()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    train, train_pca = load_split("train")
    val, val_pca = load_split("validation")
    test, test_pca = load_split("test")

    expected_sizes = {
        "train": 41998,
        "validation": 8998,
        "test": 9000,
    }

    actual_sizes = {
        "train": len(train),
        "validation": len(val),
        "test": len(test),
    }

    if actual_sizes != expected_sizes:
        raise ValueError(
            f"Unexpected split sizes: "
            f"{actual_sizes} != {expected_sizes}"
        )

    # --------------------------------------------------------
    # CROSS-SPLIT ID CHECK
    # --------------------------------------------------------

    train_ids = set(train[ID_COL])
    val_ids = set(val[ID_COL])
    test_ids = set(test[ID_COL])

    tv = train_ids & val_ids
    tt = train_ids & test_ids
    vt = val_ids & test_ids

    print(
        "\nCross-split overlap:"
    )
    print(
        f"  train/validation: {len(tv)}"
    )
    print(
        f"  train/test:       {len(tt)}"
    )
    print(
        f"  validation/test:  {len(vt)}"
    )

    if tv or tt or vt:
        raise ValueError(
            "Cross-split commit ID overlap detected"
        )

    # --------------------------------------------------------
    # LABEL CHECK
    # --------------------------------------------------------

    print("\nLabel distributions:")

    for name, df in [
        ("train", train),
        ("validation", val),
        ("test", test),
    ]:
        print(
            f"  {name}: "
            f"rows={len(df):,}, "
            f"buggy={df[TARGET].sum():,}, "
            f"rate={df[TARGET].mean():.4f}"
        )

    # --------------------------------------------------------
    # RUN A
    # --------------------------------------------------------

    all_results = []

    all_results += run_variant(
        variant_name="A_all51",
        train_df=train,
        val_df=val,
        test_df=test,
        train_pca=train_pca,
        val_pca=val_pca,
        test_pca=test_pca,
        categorical_columns=LLM_CATEGORICAL,
    )

    # --------------------------------------------------------
    # RUN B
    # --------------------------------------------------------

    all_results += run_variant(
        variant_name="B_conf_margin14",
        train_df=train,
        val_df=val,
        test_df=test,
        train_pca=train_pca,
        val_pca=val_pca,
        test_pca=test_pca,
        categorical_columns=[],
    )

    # --------------------------------------------------------
    # RUN C
    #
    # Remove ONLY the SECURITY categorical dimension.
    # Keep risk and all 14 confidence/margin features.
    # --------------------------------------------------------

    reduced_categories = [
        "intent",
        "change",
        "risk",
        "complexity",
        "scope",
        "test",
    ]

    all_results += run_variant(
        variant_name="C_reduced_no_security",
        train_df=train,
        val_df=val,
        test_df=test,
        train_pca=train_pca,
        val_pca=val_pca,
        test_pca=test_pca,
        categorical_columns=reduced_categories,
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results = pd.DataFrame(
        all_results
    )

    results.to_csv(
        OUT_DIR / "all_variant_metrics.csv",
        index=False,
    )

    test_comparison = results[
        results["split"] == "test"
    ][
        [
            "variant",
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "mcc",
            "roc_auc",
            "pr_auc",
        ]
    ].sort_values(
        ["model", "variant"]
    )

    validation_comparison = results[
        results["split"] == "validation"
    ][
        [
            "variant",
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "mcc",
            "roc_auc",
            "pr_auc",
        ]
    ].sort_values(
        ["model", "variant"]
    )

    test_comparison.to_csv(
        OUT_DIR / "test_comparison.csv",
        index=False,
    )

    validation_comparison.to_csv(
        OUT_DIR / "validation_comparison.csv",
        index=False,
    )

    metadata = {
        "experiment": (
            "LLM redundancy-reduced feature fusion"
        ),
        "jit_features": 12,
        "codebert_features": 384,
        "codebert_storage": (
            "embedding column in *_pca384.csv"
        ),
        "llm_variants": {
            "A_all51": (
                "37 categorical one-hot + "
                "14 confidence/margin"
            ),
            "B_conf_margin14": (
                "14 confidence/margin only"
            ),
            "C_reduced_no_security": (
                "intent/change/risk/complexity/"
                "scope/test categorical one-hot + "
                "14 confidence/margin; "
                "security categorical dimension removed"
            ),
        },
        "split": (
            "existing project-wise chronological "
            "70/15/15"
        ),
        "pca": "existing train-only PCA",
        "llm_ohe": "fit on train only",
        "llm_imputation": "median fit on train only",
        "resampling": False,
        "random_split": False,
        "test_used_for_selection": False,
        "threshold": 0.5,
        "models": [
            "RandomForest",
            "XGBoost",
            "LightGBM",
        ],
    }

    with open(
        OUT_DIR / "experiment_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # FINAL TABLE
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("FINAL TEST COMPARISON")
    print("=" * 78)

    print(
        test_comparison.to_string(
            index=False
        )
    )

    print("\nResults saved to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
