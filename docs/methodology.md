# Methodology

## Overview

This project implements a **Hybrid Feature Fusion Framework for Just-in-Time (JIT) Software Defect Prediction**, combining:

1. **Traditional JIT Metrics:** Process, churn, ownership, and developer experience features.
2. **CodeBERT Semantic Features:** Dense semantic representations of commit messages and diffs (PCA-reduced to 384 dimensions).
3. **LLM Reasoning Features:** Structured semantic reasoning attributes extracted via large language models.

The framework benchmarks the contribution of each feature source individually and evaluates multiple fusion paradigms (Early Fusion, Top-K Feature Selection, and Late Fusion Stacking).

---

# Dataset Preparation & Sampling

## Dataset Characteristics
* **Language Focus:** Apache Java ecosystem commits (extensible to multi-language datasets).
* **Binary Labels:**
  * `0` $\rightarrow$ Non-buggy (Clean commit)
  * `1` $\rightarrow$ Buggy (Defect-inducing commit identified via SZZ)
* **Temporal Scope:** Commits from 2018 to 2026.

## Stratified Sampling
To create a reproducible and computationally efficient experimental sample:
* **Stratification Columns:** `project` $\times$ `buggy`
* **Target Size:** 10,000 commits (`results/data/java_only_frozen_sample_dataset.csv`)
* **Random Seed:** Locked to `random_state=42` with stable `mergesort` on `author_date`.

---

# Chronological Evaluation Protocol

Real-world defect prediction is inherently non-stationary due to evolving codebases, developer churn, and defect prevalence shifts over time. To prevent lookahead bias and overoptimistic random split inflation, we enforce a strict **3-way chronological partition**:

```
[--- Train: 6,000 commits (60%) ---] [--- Validation: 2,000 (20%) ---] [--- Untouched Final Test: 2,000 (20%) ---]
     2018-01-01 to 2019-03-22               2019-03-23 to 2019-07-31               2019-07-31 to 2019-12-26
     Buggy Prevalence: 18.82%               Buggy Prevalence: 11.25%               Buggy Prevalence: 7.25%
```

### Partition Roles:
1. **Train Split (6,000 rows):** Used exclusively for model parameter estimation (gradient boosting trees, Random Forest) and supervised feature selection (e.g., Mutual Information).
2. **Validation Split (2,000 rows):** Used exclusively for:
   * Finding the optimal decision threshold $\tau^*$ that maximizes F1 or MCC.
   * Fitting Platt scaling probability calibrators.
   * Training stacking ensemble meta-learners.
   * Tuning blend weights ($\alpha$).
3. **Untouched Final Test Split (2,000 rows):** Kept completely blind until final evaluation. Evaluated exactly once per model configuration to ensure zero data leakage.

---

# Feature Sources & Engineering

## 1. Traditional JIT Metrics (13 Features)
Extracted from Git commit metadata:
* **Diffusion:** `nf` (number of modified files), `ns` (number of modified subsystems), `nd` (number of modified directories), `ent` (entropy / distribution of changes across files).
* **Size:** `la` (lines added), `ld` (lines deleted), `nuc` (number of unique modified files).
* **Purpose:** `fix` (whether the commit is a bug-fixing commit).
* **History:** `ndev` (number of previous developers modifying the files), `age` (mean time since last modification).
* **Experience:** `aexp` (author experience in lines/commits), `arexp` (recent author experience), `asexp` (author experience in the subsystem).

## 2. CodeBERT Semantic Features (384 Features)
* **Model:** Microsoft `codebert-base` (pre-trained RoBERTa architecture for programming languages).
* **Input Text:** Combined commit message and unified diff tokens: `[CLS] message [SEP] diff [SEP]` (max length 256).
* **Embedding Vector:** Extracted 768-dimensional `[CLS]` token representation.
* **Dimensionality Reduction:** Unsupervised Principal Component Analysis (PCA) retaining the top 384 components (`pca_1` .. `pca_384`), preserving primary semantic variance while reducing model complexity.

## 3. LLM Reasoning Features (Structured Attributes)
Semantic reasoning features extracted via structured prompting:
* `Intent`, `Change Type`, `Risk Level`, `Complexity`, `Scope`, `Test Impact`, `Security Risk`, `Confidence Score`.

---

# Model Training & Fusion Architectures

## Base Classifiers
* **XGBoost (`XGBClassifier`):** Primary gradient-boosted decision tree algorithm configured with dynamic class weighting ($\text{scale\_pos\_weight} = \text{negative}/\text{positive}$).
* **Random Forest (`RandomForestClassifier`):** Bagging ensemble baseline.

## Fusion Paradigms
1. **Early Fusion (Feature-Level Concatenation):**
   * Combines all 13 JIT features with 384 CodeBERT PCA components into a unified 397-dimensional feature vector.
2. **Top-K Feature Selection Fusion:**
   * Combines 13 JIT features with the top 25 or 50 CodeBERT PCA components to reduce feature dilution.
   * Supervised Mutual Information feature selection computed on the training set.
3. **Late Fusion / Stacking Ensemble (Decision-Level Fusion):**
   * Base Model 1 trains on JIT features $\rightarrow \hat{P}_{\text{JIT}}$.
   * Base Model 2 trains on CodeBERT features $\rightarrow \hat{P}_{\text{CodeBERT}}$.
   * Meta-Learner (Logistic Regression) trains on validation prediction probabilities $[\hat{P}_{\text{JIT}}, \hat{P}_{\text{CodeBERT}}]$ to produce final predictions.
4. **Probability Blending:**
   * Weighted probability blend: $\hat{P}_{\text{fused}} = \alpha \cdot \hat{P}_{\text{JIT}} + (1-\alpha) \cdot \hat{P}_{\text{CodeBERT}}$, where $\alpha$ is tuned on validation.

---

# Evaluation Metrics & Statistical Validation

To evaluate both ranking ability (threshold-independent) and operating point quality (threshold-dependent):

* **Threshold-Independent Metrics:**
  * **ROC-AUC:** Area Under the Receiver Operating Characteristic curve.
  * **PR-AUC (Average Precision):** Area Under the Precision-Recall curve (the most critical metric under class imbalance).
* **Threshold-Dependent Metrics:**
  * **Precision**, **Recall**, **F1-Score**, **Matthews Correlation Coefficient (MCC)**, and **Confusion Matrix (TP, FP, FN, TN)**.
* **Calibration Metrics:**
  * **Brier Score** and **Expected Calibration Error (ECE)**.
* **Statistical Significance Testing:**
  * **Non-parametric Bootstrap Resampling (2,000 iterations)** on the untouched test set to estimate $95\%$ confidence intervals for $\Delta \text{PR-AUC}$ and $\Delta \text{F1}$.
