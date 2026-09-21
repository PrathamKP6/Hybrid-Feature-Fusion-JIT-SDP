# Feature Descriptions

This document describes the feature representations used throughout the controlled multilingual JIT software defect prediction experiments.

---

# Traditional JIT Metrics

The final JIT representation contains **12 traditional process and code-churn metrics**.

| Feature | Description |
| ------- | ----------- |
| LA | Lines Added |
| LD | Lines Deleted |
| NF | Number of Files Modified |
| NS | Number of Subsystems Modified |
| ND | Number of Directories Modified |
| Entropy | Distribution of code changes across files |
| NDEV | Number of Developers |
| AGE | Average time since last modification |
| NUC | Number of Unique Changes |
| AEXP | Author Experience |
| AREXP | Recent Author Experience |
| ASEXP | Subsystem Experience |

These 12 features are used as the traditional JIT-SDP representation in Experiments 1, 4, 5, 7, 8, and 9.

**Note:** `FIX` is not part of the final 12-feature JIT representation used in the frozen experiments.

---

# CodeBERT Features

## Original CodeBERT Representation

Each commit is represented using a **768-dimensional CodeBERT embedding** derived from the code-change representation.

The original 768-dimensional embedding is not used directly in the final experiments.

## PCA-Reduced CodeBERT Representation

PCA is used to reduce the CodeBERT representation from 768 dimensions to **384 dimensions**.

| Feature | Description |
| ------- | ----------- |
| codebert_pca_1 | PCA-reduced CodeBERT component 1 |
| codebert_pca_2 | PCA-reduced CodeBERT component 2 |
| ... | ... |
| codebert_pca_384 | PCA-reduced CodeBERT component 384 |

### PCA Methodology

- PCA is fitted **only on the training split**.
- The fitted PCA transformation is then applied unchanged to validation and test data.
- The 384 components retain approximately **99.43% of the variance** observed in the training embeddings.
- No PCA fitting is performed using validation or test data.

The full 384-dimensional representation is used in the CodeBERT-only and full fusion experiments.

Reduced subsets of the same representation are also used:

- First 25 PCA components in Experiments 4B1 and 7B.
- First 50 PCA components in Experiments 4B2 and 7C.
- Train-only Mutual Information selection in Experiments 4C and 7D.

---

# LLM Semantic Features

The LLM representation is generated using structured semantic analysis of commit changes.

Seven semantic dimensions are extracted:

1. Intent
2. Change
3. Risk
4. Complexity
5. Scope
6. Test
7. Security

For each semantic dimension, the LLM provides:

- One categorical semantic label
- One confidence value
- One margin value

The complete model-ready representation contains:

**51 features = 37 one-hot categorical features + 14 confidence/margin features**

Categorical encoding is fitted **on the training data only** using one-hot encoding with unknown categories handled without failure.

The categorical labels are **not ordinal encoded** because their categories do not represent a natural numerical ordering.

---

# Intent Features

Intent describes the primary purpose of the software change.

Possible semantic categories include:

| Category | Description |
| -------- | ----------- |
| BF | Bug Fix |
| FT | Feature Addition |
| RF | Refactoring |
| DC | Documentation Change |
| TS | Testing |
| BL | Build / Dependency |
| PF | Performance |
| SC | Security |
| OT | Other |

After train-only one-hot encoding, the observed categories become model features such as:

| Feature | Description |
| ------- | ----------- |
| intent_BF | Commit classified as bug-fix intent |
| intent_FT | Commit classified as feature-addition intent |
| intent_RF | Commit classified as refactoring intent |
| intent_DC | Commit classified as documentation intent |
| intent_TS | Commit classified as testing intent |
| intent_BL | Commit classified as build/dependency intent |
| intent_PF | Commit classified as performance-related intent |
| intent_SC | Commit classified as security-related intent |
| intent_OT | Commit classified as another intent |

---

# Change Features

Change describes the primary type of modification performed by the commit.

Possible categories include:

| Category | Description |
| -------- | ----------- |
| LG | Logic Change |
| AP | API Change |
| DA | Data Manipulation |
| CF | Configuration Change |
| UI | User Interface Change |
| DP | Dependency Change |
| TS | Test Change |
| DC | Documentation Change |
| BL | Build / Release Change |
| OT | Other Change |

After train-only one-hot encoding, these become model features such as:

| Feature | Description |
| ------- | ----------- |
| change_LG | Logic-related change |
| change_AP | API-related change |
| change_DA | Data-related change |
| change_CF | Configuration-related change |
| change_UI | User-interface change |
| change_DP | Dependency-related change |
| change_TS | Test-related change |
| change_DC | Documentation change |
| change_BL | Build/release change |
| change_OT | Other change |

---

# Risk Features

Risk describes the estimated defect risk associated with the change.

Possible categories:

| Category | Description |
| -------- | ----------- |
| L | Low Risk |
| M | Medium Risk |
| H | High Risk |

After one-hot encoding:

| Feature | Description |
| ------- | ----------- |
| risk_L | Commit classified as low risk |
| risk_M | Commit classified as medium risk |
| risk_H | Commit classified as high risk |

**Note:** Risk categories are treated as categorical variables. They are **not encoded as ordinal values** such as low=0, medium=1, high=2.

---

# Complexity Features

Complexity describes the estimated implementation complexity of the change.

Possible categories:

| Category | Description |
| -------- | ----------- |
| L | Low Complexity |
| M | Medium Complexity |
| H | High Complexity |

After one-hot encoding:

| Feature | Description |
| ------- | ----------- |
| complexity_L | Low-complexity change |
| complexity_M | Medium-complexity change |
| complexity_H | High-complexity change |

---

# Scope Features

Scope describes how broadly the modification affects the software system.

Possible categories:

| Category | Description |
| -------- | ----------- |
| L | Local |
| F | Focused |
| M | Multi-component |
| C | Cross-cutting |

After one-hot encoding:

| Feature | Description |
| ------- | ----------- |
| scope_L | Local change |
| scope_F | Focused change |
| scope_M | Multi-component change |
| scope_C | Cross-cutting change |

---

# Test Features

The test dimension describes the estimated testing implication or testing level associated with the change.

Possible categories:

| Category | Description |
| -------- | ----------- |
| L | Low |
| M | Medium |
| H | High |
| N | None |

After one-hot encoding:

| Feature | Description |
| ------- | ----------- |
| test_L | Low testing implication |
| test_M | Medium testing implication |
| test_H | High testing implication |
| test_N | No testing implication |

---

# Security Features

The security dimension describes the estimated security relevance or risk of the change.

Possible categories:

| Category | Description |
| -------- | ----------- |
| N | None |
| L | Low |
| M | Medium |
| H | High |

After one-hot encoding:

| Feature | Description |
| ------- | ----------- |
| security_N | No security relevance |
| security_L | Low security relevance |
| security_M | Medium security relevance |
| security_H | High security relevance |

---

# LLM Confidence Features

For each of the seven semantic dimensions, the LLM provides a confidence value indicating the confidence associated with the predicted semantic classification.

The seven confidence features are:

| Feature | Description |
| ------- | ----------- |
| intent_confidence | Confidence in the predicted intent |
| change_confidence | Confidence in the predicted change type |
| risk_confidence | Confidence in the predicted risk level |
| complexity_confidence | Confidence in the predicted complexity |
| scope_confidence | Confidence in the predicted scope |
| test_confidence | Confidence in the predicted testing category |
| security_confidence | Confidence in the predicted security category |

These are continuous numerical features.

---

# LLM Margin Features

For each semantic dimension, the LLM also provides a margin value representing the separation between the selected semantic category and competing candidate categories.

The seven margin features are:

| Feature | Description |
| ------- | ----------- |
| intent_margin | Prediction margin for intent |
| change_margin | Prediction margin for change type |
| risk_margin | Prediction margin for risk |
| complexity_margin | Prediction margin for complexity |
| scope_margin | Prediction margin for scope |
| test_margin | Prediction margin for testing category |
| security_margin | Prediction margin for security category |

These are continuous numerical features.

---

# Complete LLM Feature Representation

The complete LLM representation contains:

| Feature Group | Number of Features |
| ------------- | ------------------: |
| One-hot categorical features | 37 |
| Confidence features | 7 |
| Margin features | 7 |
| **Total** | **51** |

The exact number of categorical one-hot features is determined by the categories observed in the training split.

All preprocessing is fitted on the training data only.

---

# Experiment 8 — Reduced LLM Representation

Experiment 8 evaluates whether the LLM representation can be compressed while retaining predictive information.

Only the **14 confidence and margin features** are retained:

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

**51 → 14 LLM features**

This represents a **72.55% reduction** in LLM feature dimensionality.

The complete Experiment 8 model representation is:

**12 JIT + 384 CodeBERT + 14 LLM = 410 features**

---

# Experiment 9 — Redundancy-Aware LLM Feature Selection

Experiment 9 evaluates an alternative 14-feature LLM representation selected using train-only relevance and redundancy criteria.

### Selection Procedure

1. Start with all 51 model-ready LLM features.
2. Compute Mutual Information with the buggy label using training data only.
3. Apply Pearson correlation-based redundancy filtering using `|r| >= 0.90`.
4. When highly correlated features are encountered, retain the feature with higher training Mutual Information.
5. The 51 features are reduced to 44 non-redundant candidates.
6. Select the top 14 remaining features according to training Mutual Information.

### Selected Features

| Rank | Feature |
| ---: | ------- |
| 1 | complexity_margin |
| 2 | scope_M |
| 3 | risk_confidence |
| 4 | scope_margin |
| 5 | complexity_M |
| 6 | test_M |
| 7 | test_confidence |
| 8 | security_margin |
| 9 | security_N |
| 10 | change_confidence |
| 11 | risk_M |
| 12 | change_OT |
| 13 | complexity_L |
| 14 | test_L |

The final Experiment 9 representation is:

**12 JIT + 384 CodeBERT + 14 selected LLM features = 410 features**

---

# Feature Usage Summary

| Representation | JIT | CodeBERT | LLM | Total |
| -------------- | ---: | -------: | --: | ----: |
| JIT-only | 12 | 0 | 0 | 12 |
| CodeBERT-only | 0 | 384 | 0 | 384 |
| LLM-only | 0 | 0 | 51 | 51 |
| JIT + CodeBERT | 12 | 384 | 0 | 396 |
| JIT + LLM | 12 | 0 | 51 | 63 |
| CodeBERT + LLM | 0 | 384 | 51 | 435 |
| Full Fusion | 12 | 384 | 51 | 447 |
| Experiment 8 | 12 | 384 | 14 confidence/margin | 410 |
| Experiment 9 | 12 | 384 | 14 selected | 410 |

---

# Preprocessing and Leakage Controls

The following rules apply to the final experimental pipeline:

- Chronological project-wise splitting is performed before model evaluation.
- PCA is fitted using **training data only**.
- LLM categorical encoding is fitted using **training data only**.
- Numerical imputation is fitted using **training data only**.
- Mutual Information feature selection is fitted using **training data only**.
- Correlation-based redundancy filtering is performed using **training data only**.
- Ensemble weights are selected using validation data only.
- The final test set is not used for feature selection, preprocessing fitting, or ensemble-weight selection.

These controls ensure that the feature representations do not use information from future validation or test observations during model construction.