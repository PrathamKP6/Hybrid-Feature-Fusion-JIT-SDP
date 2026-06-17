# Hybrid Feature Fusion Framework for Just-in-Time Software Defect Prediction

## Overview

This project implements a modular machine learning framework for Just-in-Time (JIT) Software Defect Prediction by combining multiple feature sources:

1. Traditional JIT Change Metrics
2. CodeBERT Semantic Embeddings
3. LLM-Generated Reasoning Features

The objective is to investigate whether semantic representations and reasoning-based features improve defect prediction performance compared to traditional JIT metrics.

---

## Research Objective

Traditional JIT defect prediction relies heavily on handcrafted software change metrics.

This project explores whether combining:

- Software engineering metrics
- Semantic code representations from CodeBERT
- LLM-generated reasoning features

can improve prediction of bug-inducing commits.

---

## Dataset

### Source

Multi-language commit dataset containing:

- Java
- Python
- C++
- Additional programming languages

### Dataset Statistics

| Property | Description |
|-----------|-------------|
| Total Commits | 100,000+ |
| Languages | Multi-language |
| Labels | Binary |
| Buggy Label | 1 |
| Non-Buggy Label | 0 |
| Timestamp Available | Yes |

---

## Sampling Strategy

To reduce computational requirements while preserving data distribution:

### Filtering

Only commits from:

```text
2018 - 2026
```

are considered.

### Stratified Sampling

Sampling is performed using:

```text
Language × Bug Label
```

stratification.

### Sample Size

```text
~10,000 commits
```

### Purpose

Preserves:

- Language distribution
- Bug/non-bug ratio
- Dataset diversity

---

## Temporal Evaluation Strategy

To simulate real-world software evolution:

### Chronological Ordering

Commits are sorted by timestamp.

### Split Strategy

```text
First 80%  → Training Set
Last 20%   → Testing Set
```

### Restrictions

- No random splitting
- No data shuffling
- Future commits never appear in training data

---

# Feature Sources

---

## 1. Traditional JIT Metrics

The following software change metrics are used:

| Feature |
|----------|
| LA |
| LD |
| NF |
| NS |
| ND |
| Entropy |
| NDEV |
| AGE |
| NUC |
| AEXP |
| AREXP |
| ASEXP |

---

## 2. CodeBERT Features

### Generation

CodeBERT embeddings are extracted from:

- Commit code changes
- Code diffs

### Dimensionality Reduction

Original dimensions:

```text
768
```

Reduced dimensions:

```text
384
```

Method:

```text
Principal Component Analysis (PCA)
```

---

## 3. LLM Reasoning Features

Generated using an external LLM pipeline.

Features capture:

- Change intent
- Risk assessment
- Semantic understanding
- Developer reasoning signals

Reasoning features are converted into numerical representations and merged using:

```text
commit_id
```

---

# Experiments

---

## Experiment 1

### JIT Metrics Only

Features:

```text
JIT Metrics
```

Models:

- Random Forest
- XGBoost

---

## Experiment 2

### CodeBERT Only

Features:

```text
CodeBERT PCA Features
```

Models:

- Random Forest
- XGBoost

---

## Experiment 3

### LLM Reasoning Only

Features:

```text
LLM Reasoning Features
```

Models:

- Random Forest
- XGBoost

---

## Experiment 4

### JIT + CodeBERT

Models:

- Random Forest
- XGBoost

---

## Experiment 5

### JIT + LLM

Models:

- Random Forest
- XGBoost

---

## Experiment 6

### CodeBERT + LLM

Models:

- Random Forest
- XGBoost

---

## Experiment 7

### Full Hybrid Fusion

Features:

```text
JIT + CodeBERT + LLM
```

Models:

- Random Forest
- XGBoost
- LightGBM

---

# Evaluation Metrics

The following metrics are computed for every experiment:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Matthews Correlation Coefficient (MCC)
- Confusion Matrix

---

# Visualizations

Generated for each experiment:

### ROC Curve

Shows discrimination ability.

### Precision-Recall Curve

Useful for imbalanced datasets.

### Feature Importance

Highlights influential features.

### Model Comparison Charts

Compares all models across metrics.

---

# Project Structure

```text
HYBRID-FEATURE-FUSION/

│
├── dataset_mining_code/
│
├── embeddings_code/
│
├── embeddings_and_pca_code/
│
├── preprocessing/
│   ├── load_data.py
│   ├── clean_data.py
│   ├── sampling.py
│   └── temporal_split.py
│
├── feature_extractors/
│
├── models/
│   ├── train_random_forest.py
│   ├── train_xgboost.py
│   ├── train_lightgbm.py
│   ├── evaluate.py
│   └── model_factory.py
│
├── experiments/
│   ├── experiment_1_jit_only.py
│   ├── experiment_2_codebert_only.py
│   ├── experiment_3_reasoning_only.py
│   ├── experiment_4_jit_codebert.py
│   ├── experiment_5_jit_reasoning.py
│   ├── experiment_6_codebert_reasoning.py
│   └── experiment_7_full_fusion.py
│
├── results/
│   ├── experiment_1/
│   ├── experiment_2/
│   ├── experiment_3/
│   ├── experiment_4/
│   ├── experiment_5/
│   ├── experiment_6/
│   └── experiment_7/
│
└── README.md
```

---

# Reproducibility Workflow

For every experiment:

1. Load dataset
2. Apply filtering
3. Perform stratified sampling
4. Merge required feature sources
5. Apply temporal split
6. Train models
7. Evaluate performance
8. Save outputs

---

# Saved Outputs

Each experiment stores:

### Models

```text
results/experiment_x/models/
```

### Metrics

```text
results/experiment_x/metrics/
```

### Predictions

```text
results/experiment_x/predictions/
```

### Visualizations

```text
results/experiment_x/plots/
```

---

# Future Extensions

The framework is designed to support:

- Additional embeddings
- Graph-based representations
- Transformer models
- Deep learning classifiers
- New reasoning feature generators
- Additional defect prediction datasets

with minimal code modifications.

---

# License

Academic Research Project