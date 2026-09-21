# Project Guide
## Hybrid Feature Fusion Framework for Just-in-Time Software Defect Prediction

### Research Objective

Build a modular machine learning framework for software defect prediction using three complementary feature sources:

1. Traditional JIT defect prediction metrics
2. Precomputed CodeBERT embeddings
3. LLM-generated reasoning features

The framework should support independent evaluation of each feature source, pairwise fusion, and full three-way feature fusion while keeping preprocessing, model training, evaluation, and experiment configuration modular and reproducible.

---

# Dataset Characteristics

The final research dataset contains:

- 59,996 commits
- Multiple programming languages:
  - Java
  - C++
  - Python
- Binary defect classification:
  - `0` = Non-buggy
  - `1` = Buggy
- Imbalanced class distribution
- Imbalanced language distribution
- Commit timestamps available

The canonical SZZ-derived label is used as the defect target.

## Final Dataset

The main dataset is:

```text
data/final_multilanguage_szz_buggy_384d.csv
```

The original canonical dataset is:

```text
data/final_multilanguage_59996_szz_buggy.csv
```

---

# Dataset Preparation

## Language Distribution

The final dataset is approximately distributed as:

| Language | Approximate Share | Approximate Commits |
|---|---:|---:|
| Java | 60% | 36,000 |
| C++ | 20% | 12,000 |
| Python | 20% | 11,996 |
| **Total** | **100%** | **59,996** |

The final dataset preserves the multilingual nature of the project.

---

# Temporal Evaluation Strategy

The final evaluation protocol uses **project-wise chronological splitting**.

Random splitting and random shuffling are not used.

For every project:

1. Parse the commit timestamp.
2. Sort commits chronologically.
3. Use a deterministic source-row ordering as a tie-breaker when timestamps are equal.
4. Assign:
   - Oldest 70% → Training
   - Next 15% → Validation
   - Newest 15% → Test

## Final Split

```text
Training:   41,998 commits
Validation:  8,998 commits
Test:        9,000 commits
Total:      59,996 commits
```

The chronological split is intended to simulate future-defect prediction: models are trained on earlier commits and evaluated on later commits.

There is no random train/test split.

There is no random shuffling.

---

# Leakage Prevention

All data-dependent preprocessing must be fitted using the training split only.

This includes:

- PCA
- One-hot encoding
- Numerical imputation
- Mutual Information feature selection
- Correlation-based feature filtering

Validation data is used for model/configuration selection where required.

The final test set is not used to select features, preprocessing parameters, models, or ensemble weights.

---

# Feature Sources

## 1. Traditional JIT Features

The final JIT representation contains 12 features:

```text
la
ld
nf
ns
nd
ent
ndev
age
nuc
aexp
arexp
asexp
```

### Feature Descriptions

| Feature | Description |
|---|---|
| `la` | Lines added |
| `ld` | Lines deleted |
| `nf` | Number of modified files |
| `ns` | Number of modified subsystems |
| `nd` | Number of modified directories |
| `ent` | Entropy of the change distribution |
| `ndev` | Number of developers associated with modified files |
| `age` | Average time since previous modification |
| `nuc` | Number of unique changes |
| `aexp` | Author experience |
| `arexp` | Recent author experience |
| `asexp` | Subsystem experience |

The final 12-feature JIT representation does not include `fix`.

---

# 2. CodeBERT Features

CodeBERT embeddings are generated externally and are already available.

The framework must **not** generate CodeBERT embeddings.

The framework must **not** perform Transformer inference.

The framework must **not** fit PCA during the experiments.

## Embedding Dimensions

Original CodeBERT representation:

```text
768 dimensions
```

Final PCA-reduced representation:

```text
384 dimensions
```

The final 384-dimensional embeddings were generated using a leakage-controlled PCA procedure in which PCA was fitted only on the training split.

The resulting embeddings are directly loaded by the experiment scripts.

Conceptually, the features are:

```text
codebert_pca_1
codebert_pca_2
...
codebert_pca_384
```

The embeddings are merged with the main dataset using:

```text
commit_id
```

---

# 3. LLM Reasoning Features

LLM reasoning features are generated externally.

The experiment framework must not perform LLM inference.

The reasoning output is treated as structured tabular data and merged with the canonical dataset using:

```text
commit_id
```

## Semantic Dimensions

The LLM representation contains seven semantic dimensions.

### 1. Intent

Possible categories include:

```text
bug_fix
feature_addition
feature_modification
refactoring
performance_improvement
documentation
configuration_change
test_update
dependency_update
```

### 2. Change Type

Possible categories include:

```text
control_flow
data_manipulation
api_change
exception_handling
ui_change
database_change
configuration_change
test_change
security_change
```

### 3. Risk Level

The final representation uses:

```text
low
medium
high
```

No `critical` risk category is used in the final production feature representation.

### 4. Complexity

```text
low
medium
high
```

### 5. Scope

```text
single_line
single_file
multiple_files
cross_module
```

### 6. Test Impact

The final production LLM schema represents test impact as a categorical semantic dimension.

### 7. Security Risk

The final production LLM schema represents security risk as a categorical semantic dimension.

---

# LLM Confidence and Margin Features

The production LLM representation contains confidence and margin values for the seven semantic dimensions.

## Confidence

```text
intent_confidence
change_confidence
risk_confidence
complexity_confidence
scope_confidence
test_confidence
security_confidence
```

## Margin

```text
intent_margin
change_margin
risk_margin
complexity_margin
scope_margin
test_margin
security_margin
```

The complete production LLM representation contains:

```text
37 one-hot categorical features
+
14 confidence/margin features
=
51 LLM features
```

Categorical variables are one-hot encoded.

They are not treated as ordinal numerical values.

One-hot encoding is fitted on the training split only.

---

# Required LLM Files

The LLM pipeline uses externally generated files.

## Raw Reasoning

```text
llm_reasoning_raw.csv
```

Expected conceptual contents include:

```text
commit_id
reasoning_text
```

Raw natural-language reasoning is not used directly as a model feature in the final experiments.

## Structured Reasoning Features

```text
reasoning_features.csv
```

This contains the structured LLM features used by the machine-learning pipeline.

The final experiments treat these features as structured numerical/categorical model inputs after the appropriate train-only encoding.

---

# Experiments

## Experiment 1 — JIT Only

### Features

```text
12 JIT features
```

### Models

- Random Forest
- XGBoost

### Purpose

Establish the traditional JIT defect-prediction baseline.

---

# Experiment 2 — CodeBERT Only

### Features

```text
384 PCA-reduced CodeBERT features
```

### Models

- Random Forest
- XGBoost

### Purpose

Evaluate the semantic CodeBERT representation independently.

---

# Experiment 3 — LLM Only

### Features

```text
51 LLM features
```

Composition:

```text
37 one-hot categorical features
+
14 confidence/margin features
```

### Models

- Random Forest
- XGBoost

### Purpose

Evaluate the structured LLM representation independently.

---

# Experiment 4 — JIT + CodeBERT

### Features

```text
12 JIT
+
384 CodeBERT
=
396 features
```

### Models

- Random Forest
- XGBoost
- LightGBM for the later weighted fusion configuration

### Fusion Variants

Experiment 4 includes:

- Early feature fusion
- First 25 CodeBERT PCA components
- First 50 CodeBERT PCA components
- Train-only Mutual Information selection
- Probability stacking
- Weighted probability blending
- RF/XGBoost/LightGBM probability blending

---

# Experiment 5 — JIT + LLM

### Features

```text
12 JIT
+
51 LLM
=
63 features
```

### Models

- Random Forest
- XGBoost

### Purpose

Evaluate whether LLM semantic features provide information complementary to traditional JIT metrics.

---

# Experiment 6 — CodeBERT + LLM

### Features

```text
384 CodeBERT
+
51 LLM
=
435 features
```

### Models

- Random Forest
- XGBoost

### Purpose

Evaluate the combination of semantic code representations and structured LLM reasoning features.

---

# Experiment 7 — Full Three-Way Fusion

### Features

```text
12 JIT
+
384 CodeBERT
+
51 LLM
=
447 features
```

### Models

- Random Forest
- XGBoost
- LightGBM

### Variants

Experiment 7 includes:

```text
7A  Full 447-feature fusion
7B  JIT + first 25 CodeBERT components + LLM
7C  JIT + first 50 CodeBERT components + LLM
7D  Train-only Mutual Information feature selection
7E  Probability stacking
7F  Validation-selected weighted probability blending
```

---

# Experiment 8 — Compact LLM Representation

Experiment 8 evaluates a reduced LLM representation.

Instead of all 51 LLM features, only the 14 continuous confidence and margin features are retained.

```text
7 confidence features
+
7 margin features
=
14 LLM features
```

The resulting representation is:

```text
12 JIT
+
384 CodeBERT
+
14 LLM
=
410 features
```

This reduces the LLM representation from 51 to 14 features, a reduction of approximately 72.5%.

The experiment investigates whether comparable predictive information can be retained using the confidence and margin representation alone.

---

# Experiment 9 — Redundancy-Aware LLM Feature Selection

Experiment 9 applies a train-only feature-selection procedure to the 51 LLM features.

## Procedure

1. Calculate Mutual Information on the training split.
2. Identify highly correlated features using Pearson correlation.
3. Use:

```text
|r| >= 0.90
```

as the redundancy threshold.
4. Retain the higher-MI feature among highly correlated candidates.
5. Select the top 14 remaining LLM features using training-set Mutual Information.

The selected representation contains:

```text
12 JIT
+
384 CodeBERT
+
14 selected LLM features
=
410 features
```

The purpose is to evaluate whether generic relevance-plus-redundancy filtering improves upon the simpler confidence/margin representation.

---

# Models

## Random Forest

Recommended common configuration:

```text
n_estimators = 300
max_features = "sqrt"
class_weight = "balanced"
random_state = 42
n_jobs = -1
```

---

## XGBoost

Recommended common configuration:

```text
n_estimators = 300
max_depth = 6
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
eval_metric = "logloss"
random_state = 42
n_jobs = -1
```

Class imbalance is handled using a training-set-derived positive-class weight.

For the final training split:

```text
scale_pos_weight ≈ 2.5257
```

---

## LightGBM

Used in the later three-way fusion experiments.

Recommended configuration:

```text
n_estimators = 300
max_depth = 6
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
scale_pos_weight = 2.5257
objective = "binary"
random_state = 42
n_jobs = -1
verbosity = -1
```

No SMOTE, oversampling, or undersampling is used in the final experimental protocol.

---

# Evaluation Metrics

Every experiment should report:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Matthews Correlation Coefficient (MCC)
- Confusion Matrix

The standard classification threshold for frozen model evaluation is:

```text
0.50
```

ROC-AUC and PR-AUC should be computed from prediction probabilities rather than thresholded predictions.

PR-AUC is particularly important because the dataset is imbalanced and the buggy prevalence changes over time.

---

# Visualizations

The framework should generate:

## ROC Curves

For validation/test evaluation where applicable.

## Precision-Recall Curves

For validation/test evaluation where applicable.

## Feature Importance

For tree-based models using the appropriate model-specific importance representation.

Feature importance must be generated without using the test set to select features.

## Model Comparison Charts

Comparison plots should support:

- Accuracy
- Precision
- Recall
- F1
- MCC
- ROC-AUC
- PR-AUC

---

# Modular Implementation

The framework should use a modular structure so additional feature sources, preprocessing stages, and classifiers can be added without rewriting the experiment pipeline.

Recommended structure:

```text
Hybrid-Feature-Fusion-JIT-SDP/
│
├── preprocessing/
│   ├── load_data.py
│   ├── clean_data.py
│   ├── sampling.py
│   ├── temporal_split.py
│   └── feature_selection.py
│
├── models/
│   ├── train_random_forest.py
│   ├── train_xgboost.py
│   ├── train_lightgbm.py
│   ├── evaluate.py
│   └── model_factory.py
│
├── fusion/
│   └── feature_fusion.py
│
├── experiments/
│   ├── experiment_1_jit_only.py
│   ├── experiment_2_codebert_only.py
│   ├── experiment_3_llm_only.py
│   ├── experiment_4_jit_codebert.py
│   ├── experiment_5_jit_llm.py
│   ├── experiment_6_codebert_llm.py
│   ├── experiment_7_full_fusion.py
│   ├── experiment_8_jit_codebert_reduced_llm.py
│   └── experiment_9_redundancy_aware_llm_selection.py
│
├── utils/
│   ├── config.py
│   ├── metrics.py
│   └── logger.py
│
├── data/
├── results/
├── models/
├── docs/
└── README.md
```

---

# Implementation Requirements

The codebase must follow:

- Python 3.11+
- Type hints
- Docstrings
- Logging
- Exception handling
- Reusable modules
- Configuration-driven design
- Deterministic random seeds
- Clear output directories
- Reproducible experiment execution

Experiment scripts should contain minimal duplicated logic.

Shared functionality should be placed in reusable modules.

---

# Configuration-Driven Design

Paths, feature lists, model parameters, random seeds, split information, and experiment settings should be configurable rather than hard-coded throughout individual experiment scripts.

A central configuration module should define shared constants such as:

```text
DATA_PATH
TRAIN_PATH
VALIDATION_PATH
TEST_PATH

JIT_FEATURES
CODEBERT_FEATURES
LLM_FEATURES

RANDOM_STATE
CLASS_WEIGHT / SCALE_POS_WEIGHT

OUTPUT_DIRECTORIES
```

---

# Feature Fusion Design

The fusion module should provide reusable functions for combinations such as:

```text
JIT
CodeBERT
LLM
JIT + CodeBERT
JIT + LLM
CodeBERT + LLM
JIT + CodeBERT + LLM
```

Feature alignment must be performed using:

```text
commit_id
```

The `commit_id` column is used for alignment and verification but must not be passed to the machine-learning model as an input feature.

---

# Preprocessing Rules

## Numerical Features

Numerical missing values, if present, must be handled using statistics fitted on the training set only.

The fitted preprocessing transformation is then applied to validation and test data.

## Categorical Features

Categorical LLM variables must be one-hot encoded.

The encoder is fitted on training data only.

Unknown categories appearing later are handled without refitting the encoder.

## CodeBERT

CodeBERT embeddings are loaded directly from the precomputed PCA-reduced dataset.

Do not:

- regenerate embeddings;
- run Transformer inference;
- fit PCA;
- fit PCA on validation/test data.

## LLM

LLM reasoning features are loaded from externally generated files.

Do not run LLM inference inside the experiment scripts.

---

# Saving Results

Each experiment should save:

## Models

```text
models/
results/<experiment>/models/
```

## Metrics

```text
metrics.json
metrics_summary.csv
```

## Predictions

Save validation and test prediction probabilities and labels where applicable.

## Visualizations

Save:

```text
ROC curves
Precision-Recall curves
Feature importance plots
Model comparison plots
Confusion matrices
```

## Metadata

Where useful, save:

- feature configuration
- selected features
- preprocessing configuration
- model parameters
- split information
- random seed
- ensemble weights
- experiment name

This makes every experiment independently auditable and reproducible.

---

# Important Restrictions

The framework must **not** generate code for:

```text
CodeBERT embedding extraction
Transformer inference
PCA fitting
LLM reasoning generation
```

These components are external/precomputed inputs to the machine-learning experiments.

The experiment framework should directly load:

```text
384-dimensional PCA-reduced CodeBERT embeddings
```

and:

```text
structured LLM reasoning features
```

and merge them using:

```text
commit_id
```

---

# Reproducibility Requirements

The final framework should guarantee:

- deterministic split generation;
- fixed random seeds;
- no random train/test shuffling;
- project-wise chronological evaluation;
- training-only preprocessing;
- training-only feature selection;
- validation-only ensemble selection;
- no test-driven feature selection;
- no duplicate commit IDs across splits;
- no commit leakage between train, validation, and test;
- consistent feature ordering across experiments.

---

# Expected Research Workflow

```text
Multilingual Commit Dataset
          │
          ▼
      SZZ Labels
          │
          ▼
Project-wise Chronological Split
          │
          ├───────────────┐
          ▼               ▼
    JIT Features      CodeBERT 384-D
          │               │
          │               │
          └───────┬───────┘
                  │
                  ▼
          LLM Structured Features
                  │
                  ▼
            Feature Fusion
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
       JIT     CodeBERT     LLM
        │         │         │
        └─────┬───┴─────┬───┘
              │         │
              ▼         ▼
          Pairwise     Full
           Fusion     Fusion
              │         │
              └────┬────┘
                   ▼
          RF / XGBoost / LGBM
                   │
                   ▼
          Validation Evaluation
                   │
                   ▼
       Configuration / Ensemble Selection
                   │
                   ▼
             Frozen Model
                   │
                   ▼
             Test Evaluation
                   │
                   ▼
       Metrics + Predictions + Plots
```

---

# Final Design Principle

The framework should be designed as a reusable experimental platform rather than as a collection of independent scripts.

Adding a new feature source should require only:

1. A loader/preprocessing module.
2. A feature definition in the configuration.
3. A fusion configuration.
4. A new experiment script or experiment configuration.

Adding a new classifier should similarly require:

1. A model implementation.
2. Registration in the model factory.
3. Configuration of its hyperparameters.
4. Inclusion in the desired experiment.

This modular design allows the framework to be extended to additional semantic representations, embedding models, feature-selection methods, classifiers, and ensemble strategies without changing the core preprocessing and evaluation pipeline.
