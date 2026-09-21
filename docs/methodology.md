# Methodology

## 1. Overview

This project implements a **Hybrid Feature Fusion Framework for Multilingual Just-in-Time (JIT) Software Defect Prediction (SDP)**. The framework combines three complementary sources of information:

1. **Traditional JIT Metrics** — process, code-churn, ownership, history, and developer-experience characteristics of a software change.
2. **CodeBERT Semantic Features** — dense semantic representations of commit changes, reduced from 768 dimensions to 384 dimensions using training-only PCA.
3. **LLM Semantic Features** — structured semantic attributes extracted from commit changes, including intent, change type, risk, complexity, scope, test impact, and security relevance, together with confidence and margin information.

The experimental framework evaluates each feature source independently and then investigates multiple fusion strategies, including **early feature fusion, dimensionality reduction, mutual-information feature selection, probability stacking, and weighted probability blending**.

The final methodology is designed to evaluate whether semantic representations from CodeBERT and structured LLM-derived information provide complementary information beyond traditional JIT metrics under a realistic chronological evaluation setting.

---

# 2. Dataset Preparation

## 2.1 Dataset Characteristics

The final dataset contains **59,996 software commits** collected from Java, C++, and Python projects.

The language distribution is approximately:

| Language | Target Share | Approx. Commits |
|---|---:|---:|
| Java | 60% | 36,000 |
| C++ | 20% | 12,000 |
| Python | 20% | 11,996 |
| **Total** | **100%** | **59,996** |

The dataset contains commits spanning approximately **2016–2026**.

The projects are represented using language-specific project groups, including:

- Apache Java projects
- C++ projects
- Python projects including CPython, Django, and Flask

### Binary Defect Labels

The prediction target is a binary SZZ-based defect label:

- `0` → Non-buggy commit
- `1` → Buggy / defect-inducing commit

The final SZZ-derived labels are treated as the **canonical labels** throughout all experiments.

The final dataset contains:

- **26,568 buggy commits**
- **33,428 non-buggy commits**
- **59,996 total commits**

Duplicate commit IDs were checked and no duplicate commit IDs were present.

---

# 3. Temporal Evaluation Protocol

## 3.1 Motivation

JIT software defect prediction is inherently affected by temporal changes in software projects, including changes in development practices, project composition, developer participation, and defect prevalence.

Therefore, random train/test splitting is not used in the final experimental protocol.

Instead, the experiments use a **project-wise chronological 70/15/15 split**.

For every project:

1. Commits are ordered chronologically using the parsed author date.
2. A deterministic source-row identifier is used as a tie-breaker when timestamps are equal.
3. The oldest 70% of commits are assigned to training.
4. The following 15% are assigned to validation.
5. The newest 15% are assigned to the final test set.

This preserves the temporal direction of prediction: models are trained on earlier commits and evaluated on later commits.

---

## 3.2 Final Dataset Split

The final chronological partition contains:

| Split | Rows | Role |
|---|---:|---|
| Training | 41,998 | Model training and train-only preprocessing |
| Validation | 8,998 | Model/ensemble selection |
| Test | 9,000 | Final evaluation |
| **Total** | **59,996** | |

No commit appears in more than one split.

The temporal ordering and project-level split boundaries were explicitly validated.

---

## 3.3 Temporal Defect Prevalence

The buggy prevalence changes across the chronological splits:

| Split | Buggy Commits | Buggy Prevalence |
|---|---:|---:|
| Train | 11,912 | 28.36% |
| Validation | 2,082 | 23.14% |
| Test | 1,376 | 15.29% |

This temporal prevalence shift is retained rather than corrected through random resampling because it represents a realistic characteristic of the chronological evaluation setting.

Consequently, performance is evaluated using both threshold-dependent and threshold-independent metrics.

---

# 4. Leakage-Controlled Preprocessing

All preprocessing operations that can learn information from the data are fitted **using the training split only**.

The following operations follow this rule:

- PCA
- categorical one-hot encoding
- numerical imputation
- Mutual Information feature selection
- correlation-based redundancy filtering

The fitted transformations are then applied unchanged to validation and test data.

Commit IDs are used for dataset alignment and verification but are **never used as predictive features**.

No random train/test split or resampling is used in the final experiments.

---

# 5. Feature Sources and Engineering

## 5.1 Traditional JIT Metrics

The final traditional JIT representation contains **12 features**.

| Feature | Description |
|---|---|
| `la` | Lines added |
| `ld` | Lines deleted |
| `nf` | Number of modified files |
| `ns` | Number of modified subsystems |
| `nd` | Number of modified directories |
| `ent` | Entropy / distribution of changes across files |
| `ndev` | Number of developers associated with the modified files |
| `age` | Average time since previous modification |
| `nuc` | Number of unique changes |
| `aexp` | Author experience |
| `arexp` | Recent author experience |
| `asexp` | Subsystem experience |

The `fix` feature is **not included** in the final 12-feature JIT representation used in the frozen experiments.

These 12 features form the baseline traditional JIT representation for the JIT-only and fusion experiments.

---

# 6. CodeBERT Semantic Representation

## 6.1 Original Embeddings

Each commit is represented using a **768-dimensional CodeBERT embedding** generated from the commit's semantic information.

The original embedding dimensionality is:

**768 dimensions**

The original 768-dimensional representation is not directly used in the final fusion models.

---

## 6.2 PCA Dimensionality Reduction

Principal Component Analysis (PCA) is used to reduce the CodeBERT representation from **768 to 384 dimensions**.

The PCA procedure is leakage-controlled:

1. PCA is fitted exclusively using the training embeddings.
2. The fitted PCA transformation is applied to validation embeddings.
3. The same transformation is applied to test embeddings.
4. No validation or test embeddings are used when fitting PCA.

The 384-dimensional representation retains approximately **99.43% of the variance observed in the training embeddings**.

The resulting conceptual features are:

```text
codebert_pca_1
codebert_pca_2
...
codebert_pca_384
```

The full 384-dimensional CodeBERT representation is used in the final CodeBERT and fusion experiments.

Reduced CodeBERT representations are additionally evaluated using:

- First 25 PCA components
- First 50 PCA components
- Training-only Mutual Information-selected components

---

# 7. LLM Semantic Features

## 7.1 Structured Semantic Extraction

LLM-derived semantic information is extracted from commit changes using a structured classification approach.

The final LLM representation contains **seven semantic dimensions**:

1. Intent
2. Change
3. Risk
4. Complexity
5. Scope
6. Test
7. Security

For each semantic dimension, the extraction process produces:

- one categorical semantic label
- one confidence value
- one margin value

The production LLM dataset contains **59,996 commits** with the corresponding structured semantic attributes.

---

## 7.2 LLM Categorical Features

The seven semantic dimensions use categorical labels.

### Intent

Possible categories include:

- `BF` — Bug Fix
- `FT` — Feature Addition
- `RF` — Refactoring
- `DC` — Documentation
- `TS` — Testing
- `BL` — Build / Dependency
- `PF` — Performance
- `SC` — Security
- `OT` — Other

### Change

Possible categories include:

- `LG` — Logic
- `AP` — API
- `DA` — Data Manipulation
- `CF` — Configuration
- `UI` — User Interface
- `DP` — Dependency
- `TS` — Test
- `DC` — Documentation
- `BL` — Build / Release
- `OT` — Other

### Risk

- `L` — Low
- `M` — Medium
- `H` — High

### Complexity

- `L` — Low
- `M` — Medium
- `H` — High

### Scope

- `L` — Local
- `F` — Focused
- `M` — Multi-component
- `C` — Cross-cutting

### Test

- `L` — Low
- `M` — Medium
- `H` — High
- `N` — None

### Security

- `N` — None
- `L` — Low
- `M` — Medium
- `H` — High

Categorical variables are **one-hot encoded**.

They are not ordinal encoded because the categories do not represent a single natural numerical ordering.

---

# 8. LLM Confidence and Margin Features

For each of the seven semantic dimensions, two continuous numerical attributes are retained.

## Confidence Features

- `intent_confidence`
- `change_confidence`
- `risk_confidence`
- `complexity_confidence`
- `scope_confidence`
- `test_confidence`
- `security_confidence`

## Margin Features

- `intent_margin`
- `change_margin`
- `risk_margin`
- `complexity_margin`
- `scope_margin`
- `test_margin`
- `security_margin`

Therefore, the complete LLM model-ready representation contains:

| Feature group | Number |
|---|---:|
| One-hot categorical features | 37 |
| Confidence features | 7 |
| Margin features | 7 |
| **Total** | **51** |

The categorical encoding and numerical imputation are fitted using the training split only.

---

# 9. Feature Representations Evaluated

The experiments evaluate the following feature configurations:

| Representation | JIT | CodeBERT | LLM | Total |
|---|---:|---:|---:|---:|
| JIT-only | 12 | 0 | 0 | 12 |
| CodeBERT-only | 0 | 384 | 0 | 384 |
| LLM-only | 0 | 0 | 51 | 51 |
| JIT + CodeBERT | 12 | 384 | 0 | 396 |
| JIT + LLM | 12 | 0 | 51 | 63 |
| CodeBERT + LLM | 0 | 384 | 51 | 435 |
| Full Fusion | 12 | 384 | 51 | 447 |
| Experiment 8 | 12 | 384 | 14 | 410 |
| Experiment 9 | 12 | 384 | 14 | 410 |

---

# 10. Model Training

## 10.1 Random Forest

Random Forest is used as a bagging-based tree ensemble.

The common configuration is:

```text
n_estimators = 300
max_features = sqrt
class_weight = balanced
random_state = 42
n_jobs = -1
```

The classifier is trained using the training split only.

---

## 10.2 XGBoost

XGBoost is used as a gradient-boosted decision-tree classifier.

The common configuration is:

```text
n_estimators = 300
max_depth = 6
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
eval_metric = logloss
random_state = 42
n_jobs = -1
```

Class imbalance is handled using:

\[
\text{scale\_pos\_weight}
=
\frac{N_{\text{negative}}}{N_{\text{positive}}}
\]

For the final training split:

\[
\text{scale\_pos\_weight}
=
\frac{30086}{11912}
\approx 2.526
\]

---

## 10.3 LightGBM

LightGBM is additionally used in the later fusion experiments.

The common configuration is:

```text
n_estimators = 300
max_depth = 6
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
scale_pos_weight = 2.5257
objective = binary
random_state = 42
n_jobs = -1
verbosity = -1
```

No resampling is performed.

---

# 11. Fusion Architectures

## 11.1 Early Feature Fusion

Early fusion concatenates heterogeneous feature representations before model training.

For the complete three-way representation:

\[
12\text{ JIT}
+
384\text{ CodeBERT}
+
51\text{ LLM}
=
447\text{ features}
\]

The resulting feature vector is supplied directly to the tree-based classifier.

The principal full-fusion configuration therefore contains **447 model features**.

---

# 12. Reduced CodeBERT Fusion

To investigate whether the full CodeBERT representation introduces unnecessary dimensionality, reduced representations are also evaluated.

### 12.1 First 25 PCA Components

\[
12 + 25 = 37
\]

for JIT + CodeBERT fusion.

For three-way fusion:

\[
12 + 25 + 51 = 88
\]

### 12.2 First 50 PCA Components

\[
12 + 50 = 62
\]

for JIT + CodeBERT fusion.

For three-way fusion:

\[
12 + 50 + 51 = 113
\]

### 12.3 Mutual Information Selection

Mutual Information feature selection is performed using the training split only.

The selected features are then applied unchanged to validation and test data.

---

# 13. Late Fusion and Probability-Level Fusion

## 13.1 Probability Stacking

Late fusion combines predictions generated by independently trained base classifiers.

The general procedure is:

1. Train the base models using the training data.
2. Generate validation-set prediction probabilities.
3. Use the validation probabilities as inputs to a Logistic Regression meta-learner.
4. Train the meta-learner using validation predictions.
5. Generate base-model probabilities for the test set.
6. Apply the trained meta-learner to the test probabilities.

For example:

\[
X_{\text{meta}}
=
[
P_{\text{JIT}},
P_{\text{CodeBERT}},
P_{\text{LLM}}
]
\]

For the later three-model fusion experiments, Random Forest, XGBoost, and LightGBM prediction probabilities are combined.

---

# 14. Weighted Probability Blending

Weighted probability blending combines model probabilities using validation-selected weights.

For multiple models:

\[
P_{\text{blend}}
=
w_1P_1+w_2P_2+\cdots+w_kP_k
\]

subject to:

\[
\sum_i w_i=1
\]

The weights are selected using the validation split only.

For the three-model fusion experiments, the candidate models are:

- Random Forest
- XGBoost
- LightGBM

The validation selection procedure prioritizes:

1. F1-score
2. MCC
3. ROC-AUC
4. PR-AUC

The selected weights are then frozen before evaluating the final test set.

---

# 15. Experiment 8 — Compact LLM Representation

Experiment 8 evaluates whether the complete 51-feature LLM representation can be compressed without substantially changing predictive performance.

Only the **14 continuous confidence and margin features** are retained.

### Confidence

- `intent_confidence`
- `change_confidence`
- `risk_confidence`
- `complexity_confidence`
- `scope_confidence`
- `test_confidence`
- `security_confidence`

### Margin

- `intent_margin`
- `change_margin`
- `risk_margin`
- `complexity_margin`
- `scope_margin`
- `test_margin`
- `security_margin`

Thus:

\[
51 \rightarrow 14
\]

representing a **72.55% reduction** in LLM feature dimensionality.

The resulting fusion representation is:

\[
12 + 384 + 14 = 410
\]

features.

The purpose of this experiment is to determine whether the continuous confidence and margin information retains useful predictive information while removing the one-hot categorical representation.

---

# 16. Experiment 9 — Redundancy-Aware LLM Feature Selection

Experiment 9 evaluates an alternative compact LLM representation using training-only relevance and redundancy analysis.

The procedure is:

1. Start with all 51 model-ready LLM features.
2. Compute Mutual Information between each feature and the buggy label using training data only.
3. Perform Pearson correlation-based redundancy filtering using:

\[
|r| \geq 0.90
\]

4. For highly correlated features, retain the feature with higher training Mutual Information.
5. This reduces the candidate set from 51 to 44 non-redundant features.
6. Select the top 14 remaining features according to training Mutual Information.

The selected features are:

| Rank | Feature |
|---:|---|
| 1 | `complexity_margin` |
| 2 | `scope_M` |
| 3 | `risk_confidence` |
| 4 | `scope_margin` |
| 5 | `complexity_M` |
| 6 | `test_M` |
| 7 | `test_confidence` |
| 8 | `security_margin` |
| 9 | `security_N` |
| 10 | `change_confidence` |
| 11 | `risk_M` |
| 12 | `change_OT` |
| 13 | `complexity_L` |
| 14 | `test_L` |

The resulting model representation is:

\[
12 + 384 + 14 = 410
\]

features.

The purpose of Experiment 9 is to test whether explicit train-only relevance and redundancy filtering provides an advantage over the simpler 14-feature confidence/margin representation.

---

# 17. Evaluation Metrics

The evaluation uses both ranking-based and threshold-dependent metrics.

## 17.1 Threshold-Independent Metrics

### ROC-AUC

Receiver Operating Characteristic Area Under the Curve measures ranking performance across classification thresholds.

### PR-AUC

Average Precision is used as the Precision-Recall based ranking metric.

Because the buggy class becomes less prevalent in the later chronological test period, PR-AUC is particularly informative for evaluating positive-class retrieval under the observed class imbalance.

---

## 17.2 Threshold-Dependent Metrics

At the fixed decision threshold of:

\[
\tau = 0.5
\]

the following metrics are reported:

- Accuracy
- Precision
- Recall
- F1-score
- Matthews Correlation Coefficient (MCC)
- Confusion Matrix

The confusion matrix reports:

- True Negatives
- False Positives
- False Negatives
- True Positives

The final experiments do not use the test set to tune the classification threshold.

---

# 18. Validation-Based Ensemble Selection

The validation set is used for model-selection operations that must not use the final test set.

These include:

- ensemble weight selection
- stacking meta-model training
- comparison of alternative fusion configurations
- feature-representation selection where applicable

Once the configuration and ensemble parameters are selected, they are frozen before final test evaluation.

The final test split is therefore reserved for estimating out-of-sample performance on later commits.

---

# 19. Experimental Controls and Reproducibility

The final experimental protocol follows these controls:

- Fixed random seed: `42`
- Project-wise chronological splitting
- No random train/test splitting
- No oversampling or undersampling
- No SMOTE or synthetic resampling
- PCA fitted on training data only
- One-hot encoding fitted on training data only
- Numerical imputation fitted on training data only
- Mutual Information selection fitted on training data only
- Correlation filtering fitted on training data only
- Validation-only ensemble-weight selection
- Commit IDs used only for alignment and verification
- Final test set kept separate from model construction and feature-selection procedures

These controls are intended to prevent temporal leakage and preprocessing leakage while maintaining a reproducible experimental pipeline.

---

# 20. Experimental Comparison Structure

The experiments are organized to isolate the contribution of the different information sources.

### Experiment 1 — JIT-only

\[
12\text{ JIT features}
\]

Establishes the traditional JIT-SDP baseline.

### Experiment 2 — CodeBERT-only

\[
384\text{ CodeBERT features}
\]

Evaluates semantic representation without traditional JIT metrics.

### Experiment 3 — LLM-only

\[
51\text{ LLM features}
\]

Evaluates structured LLM semantic information independently.

### Experiment 4 — JIT + CodeBERT

Evaluates multiple fusion mechanisms between traditional JIT and CodeBERT information.

### Experiment 5 — JIT + LLM

\[
12 + 51 = 63
\]

Evaluates the contribution of LLM semantic features to traditional JIT information.

### Experiment 6 — CodeBERT + LLM

\[
384 + 51 = 435
\]

Evaluates semantic fusion without traditional JIT metrics.

### Experiment 7 — JIT + CodeBERT + LLM

\[
12 + 384 + 51 = 447
\]

Evaluates complete three-way feature fusion using Random Forest, XGBoost, and LightGBM and multiple fusion strategies.

### Experiment 8 — Compact LLM Fusion

\[
12 + 384 + 14 = 410
\]

Tests whether confidence/margin-only LLM features can provide a compact alternative to the complete 51-feature LLM representation.

### Experiment 9 — Redundancy-Aware LLM Fusion

\[
12 + 384 + 14 = 410
\]

Tests whether train-only Mutual Information and correlation-based feature selection can further improve the compact LLM representation.

---

# 21. Final Experimental Principle

The central experimental question is whether **traditional change-level JIT metrics, CodeBERT semantic representations, and structured LLM semantic information provide complementary information for chronological multilingual JIT software defect prediction**.

The methodology therefore evaluates:

1. Each feature source independently.
2. Pairwise feature combinations.
3. Full three-way fusion.
4. Reduced CodeBERT representations.
5. Decision-level fusion.
6. Compact LLM representations.
7. Redundancy-aware LLM feature selection.

All final comparisons are conducted under the same chronological evaluation protocol and leakage-control rules, allowing the contribution of each representation and fusion strategy to be examined under a consistent experimental setting.
