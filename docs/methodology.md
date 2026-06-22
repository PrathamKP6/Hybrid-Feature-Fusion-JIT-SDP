# Methodology

## Overview

This project implements a Hybrid Feature Fusion Framework for Just-in-Time (JIT) Software Defect Prediction using:

1. Traditional JIT Metrics
2. CodeBERT Semantic Features
3. LLM Reasoning Features

The framework is designed to evaluate the contribution of each feature source individually and in combination.

---

# Dataset Preparation

## Dataset Characteristics

* Multi-language commit dataset
* More than 100,000 commits
* Binary labels:

  * 0 → Non-buggy
  * 1 → Buggy

## Temporal Filtering

Only commits from:

2018–2026

are considered for experimentation.

---

# Sampling Strategy

To preserve dataset distribution while reducing computational requirements:

## Stratification

Sampling is performed using:

Language × Bug Label

## Sample Size

Approximately 10,000 commits.

## Objective

Maintain:

* Language distribution
* Bug/non-bug distribution
* Temporal consistency

---

# Temporal Evaluation

## Chronological Ordering

Commits are sorted using commit timestamps.

## Split Strategy

* First 80% → Training
* Last 20% → Testing

## Restrictions

* No random splitting
* No shuffling

This simulates real-world defect prediction scenarios.

---

# Feature Sources

## Traditional JIT Metrics

* LA
* LD
* NF
* NS
* ND
* Entropy
* NDEV
* AGE
* NUC
* AEXP
* AREXP
* ASEXP

---

## CodeBERT Features

* Precomputed embeddings
* PCA already applied
* Final dimension: 384

Merged using:
commit_id

---

## LLM Reasoning Features

Structured reasoning dimensions include:

* Intent
* Change Type
* Risk Level
* Complexity
* Scope
* Test Impact
* Security Risk
* Confidence Score

Categorical features are one-hot encoded.

---

# Experiments

## Experiment 1

JIT Metrics Only

## Experiment 2

CodeBERT Features Only

## Experiment 3

LLM Reasoning Features Only

## Experiment 4a

JIT + CodeBERT feature fusion

## Experiment 4b

JIT + CodeBERT feature selection

## Experiment 4c

Top-K PCA selection for JIT + CodeBERT fusion

## Experiment 4d

Late fusion stacking ensemble using JIT and CodeBERT meta-level fusion

## Experiment 4e

Feature group ablation comparing JIT-only, PCA-only, and fusion configurations

---

# Models

* Random Forest
* XGBoost
* LightGBM (used in current and planned hybrid experiments)

---

# Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* PR-AUC
* MCC
* Confusion Matrix
