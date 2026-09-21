# Experiment Tracking

This document maintains the complete record of defect prediction experiments under the **final controlled multilingual chronological benchmark protocol**.

## Controlled Benchmark Protocol

- **Dataset:** 59,996 commits across Java, C++, and Python.
- **Languages:** Java ≈60%, C++ ≈20%, Python ≈20%.
- **Split:** Project-wise chronological **70% Train → 15% Validation → 15% Untouched Final Test**.
- **Rows:** Train = 41,998 | Validation = 8,998 | Test = 9,000.
- **Buggy prevalence:** Train = 28.36% | Validation = 23.14% | Test = 15.29%.
- **Temporal prevalence drift:** retained as observed; no resampling was applied.
- **Random split:** None.
- **Resampling:** None.
- **Target:** canonical SZZ-derived `buggy` label.
- **JIT representation:** 12 metrics:
  `la`, `ld`, `nf`, `ns`, `nd`, `ent`, `ndev`, `age`, `nuc`, `aexp`, `arexp`, `asexp`.
- **CodeBERT representation:** 768-D embeddings reduced to 384 PCA components.
- **PCA:** fitted on **training data only**, then reused for validation/test.
- **LLM representation:** 51 model-ready features from seven semantic dimensions plus confidence/margin features.
- **LLM categorical encoding:** OneHotEncoder fitted on training data only.
- **LLM numerical preprocessing:** train-only median imputation.
- **Feature selection:** when used, fitted on training data only.
- **Ensemble weights:** selected using validation data only and frozen before final test evaluation.
- **Final test set:** used only for final evaluation.

---

# Experiment 1 — JIT Metrics Only

## Status: COMPLETED & FROZEN

### Objective

Evaluate traditional Just-In-Time process and code-churn metrics for multilingual software defect prediction.

### Features

12 JIT features:

`la`, `ld`, `nf`, `ns`, `nd`, `ent`, `ndev`, `age`, `nuc`, `aexp`, `arexp`, `asexp`

### Final Test Results

| Model | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.7402 | 0.3178 | 0.6097 | 0.4178 | 0.2952 | 0.7681 | 0.3651 |
| XGBoost | 0.6891 | 0.2949 | 0.7427 | 0.4221 | 0.3122 | 0.7806 | 0.3802 |

### Observations

- JIT metrics provide a strong traditional baseline under chronological evaluation.
- XGBoost provides higher recall and ranking metrics than Random Forest on the final test set.
- The large train-to-test prevalence shift is retained rather than corrected through resampling.

---

# Experiment 2 — CodeBERT Semantic Features Only

## Status: COMPLETED & FROZEN

### Objective

Evaluate semantic code representations using PCA-reduced CodeBERT embeddings without JIT or LLM features.

### Features

384 PCA components:

`codebert_pca_1` ... `codebert_pca_384`

### Final Test Results

| Model | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.8140 | 0.3307 | 0.2115 | 0.2580 | 0.1626 | 0.6987 | 0.2669 |
| XGBoost | 0.6744 | 0.2584 | 0.6039 | 0.3619 | 0.2186 | 0.7134 | 0.2783 |

### Observations

- CodeBERT contains measurable defect-related signal but is weaker than the JIT-only representation on ranking metrics.
- The two models exhibit different threshold behavior, illustrating why both threshold metrics and ranking metrics are reported.
- CodeBERT is therefore evaluated primarily as a complementary semantic representation in subsequent fusion experiments.

---

# Experiment 3 — LLM Reasoning Features Only

## Status: COMPLETED & FROZEN

### Objective

Evaluate structured LLM-derived semantic features independently of JIT and CodeBERT representations.

### LLM Representation

Seven semantic dimensions:

- `intent`
- `change`
- `risk`
- `complexity`
- `scope`
- `test`
- `security`

Each dimension contributes a categorical prediction, confidence, and margin, giving **51 model-ready features** after train-only one-hot encoding.

### Final Test Results

| Model | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.6881 | 0.2510 | 0.5240 | 0.3394 | 0.1866 | 0.6686 | 0.2401 |
| XGBoost | 0.6364 | 0.2417 | 0.6446 | 0.3516 | 0.2048 | 0.6697 | 0.2425 |

### Observations

- The LLM representation contains genuine defect-related information.
- Diagnostic analysis showed particularly strong univariate signal in confidence/margin features.
- Considerable redundancy exists among the 51 LLM features, motivating Experiments 8 and 9.

---

# Experiment 4 — JIT + CodeBERT Hybrid Feature Fusion

## Status: COMPLETED, AUDITED & FROZEN

### Objective

Evaluate whether combining traditional JIT metrics with semantic CodeBERT representations improves multilingual chronological JIT-SDP.

### Variants

- **4A:** 12 JIT + 384 CodeBERT = 396 features.
- **4B1:** 12 JIT + first 25 PCA components = 37 features.
- **4B2:** 12 JIT + first 50 PCA components = 62 features.
- **4C:** Mutual Information top-25 features from the 396-feature space.
- **4D:** Probability stacking using validation predictions and Logistic Regression.
- **4E:** Validation-selected weighted probability blend of JIT and CodeBERT models.
- **4F:** Weighted probability blend of JIT+CodeBERT RF, XGBoost, and LightGBM.

### Final Test Results — 4A–4E

| Variant / Model | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4A RF | 0.7788 | 0.3524 | 0.5334 | 0.4244 | 0.3042 | 0.7772 | 0.3676 |
| 4A XGBoost | 0.7162 | 0.3193 | 0.7565 | 0.4491 | 0.3485 | 0.8037 | 0.4038 |
| 4B1 RF | 0.7593 | 0.3470 | 0.6512 | 0.4528 | 0.3421 | 0.7965 | 0.4003 |
| 4B1 XGBoost | 0.7088 | 0.3150 | 0.7703 | 0.4472 | 0.3481 | 0.8029 | 0.4049 |
| 4B2 RF | 0.7628 | 0.3525 | 0.6592 | 0.4594 | 0.3510 | 0.7957 | 0.3952 |
| 4B2 XGBoost | 0.7094 | 0.3166 | 0.7769 | 0.4498 | 0.3524 | 0.8028 | 0.3948 |
| 4C RF | 0.7561 | 0.3395 | 0.6294 | 0.4410 | 0.3261 | 0.7881 | 0.3806 |
| 4C XGBoost | 0.6994 | 0.3067 | 0.7660 | 0.4380 | 0.3359 | 0.7916 | 0.3916 |
| 4D Stacking | 0.6819 | 0.2962 | 0.7856 | 0.4302 | 0.3289 | 0.7898 | 0.3766 |
| 4E RF Blend | 0.7469 | 0.3237 | 0.6017 | 0.4209 | 0.2991 | 0.7706 | 0.3699 |
| 4E XGB Blend | 0.7079 | 0.3109 | 0.7485 | 0.4393 | 0.3351 | 0.7896 | 0.3812 |

### 4F — Three-Model Weighted Fusion

Validation-selected weights:

- RF = 0.35
- XGBoost = 0.30
- LightGBM = 0.35

| Model / Blend | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RF | 0.7788 | 0.3524 | 0.5334 | 0.4244 | 0.3042 | 0.7772 | 0.3676 |
| XGBoost | 0.7162 | 0.3193 | 0.7565 | 0.4491 | 0.3485 | 0.8037 | 0.4038 |
| LightGBM | 0.7062 | 0.3155 | 0.7878 | 0.4505 | 0.3550 | 0.8064 | 0.4068 |
| **Weighted Blend** | **0.7301** | **0.3290** | **0.7362** | **0.4548** | **0.3529** | **0.8054** | **0.4043** |

### Observations

- 4F establishes the main JIT + CodeBERT fusion reference for later LLM experiments.
- The fusion results demonstrate that semantic CodeBERT information can be combined with JIT metrics without requiring all 384 dimensions in smaller variants.
- No claim is made that every fusion architecture improves every metric; variants show different trade-offs.

---

# Experiment 5 — JIT + LLM Feature Fusion

## Status: COMPLETED & FROZEN

### Objective

Evaluate the full 51-feature LLM representation together with the 12 JIT metrics.

### Feature Space

12 JIT + 51 LLM = **63 features**

### Final Test Results

| Model | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.7394 | 0.3161 | 0.6054 | 0.4154 | 0.2919 | 0.7692 | 0.3544 |
| XGBoost | 0.6886 | 0.2949 | 0.7456 | 0.4227 | 0.3133 | 0.7780 | 0.3812 |

### Observations

The LLM representation adds semantic information to JIT metrics, but the full 51-feature representation is not by itself sufficient to establish an improvement over the stronger JIT + CodeBERT fusion reference.

---

# Experiment 6 — CodeBERT + LLM Feature Fusion

## Status: COMPLETED & FROZEN

### Objective

Evaluate semantic fusion without traditional JIT metrics.

### Feature Space

384 CodeBERT + 51 LLM = **435 features**

### Final Test Results

| Model | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.7610 | 0.3171 | 0.4884 | 0.3845 | 0.2533 | 0.7330 | 0.3045 |
| XGBoost | 0.6987 | 0.2934 | 0.6897 | 0.4117 | 0.2925 | 0.7626 | 0.3389 |

### Observations

CodeBERT + LLM captures complementary semantic information, but the absence of JIT metrics changes the performance profile relative to the full three-way fusion experiments.

---

# Experiment 7 — JIT + CodeBERT + LLM Full Fusion

## Status: COMPLETED & FROZEN

### Objective

Evaluate full three-way feature fusion using all available model representations.

### Feature Space

12 JIT + 384 CodeBERT + 51 LLM = **447 features**

### Variants

- **7A:** Full 447-feature early fusion.
- **7B:** 12 JIT + first 25 CodeBERT PCA + 51 LLM = 88 features.
- **7C:** 12 JIT + first 50 CodeBERT PCA + 51 LLM = 113 features.
- **7D:** Train-only Mutual Information top-25 selection from all 447 features.
- **7E:** Probability stacking using RF, XGBoost, and LightGBM.
- **7F:** Validation-selected weighted probability blend of RF, XGBoost, and LightGBM.

### Final Test Results

| Variant | Model | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7A | RF | 0.7633 | 0.3403 | 0.5836 | 0.4299 | 0.3104 | 0.7736 | 0.3593 |
| 7A | XGBoost | 0.7138 | 0.3176 | 0.7594 | 0.4479 | 0.3475 | 0.8019 | 0.3985 |
| 7A | LightGBM | 0.7022 | 0.3118 | 0.7849 | 0.4463 | 0.3492 | 0.8028 | 0.3984 |
| 7B | RF | 0.7582 | 0.3444 | 0.6432 | 0.4486 | 0.3364 | 0.7900 | 0.3821 |
| 7B | XGBoost | 0.7019 | 0.3095 | 0.7718 | 0.4419 | 0.3416 | 0.7988 | 0.3932 |
| 7B | LightGBM | 0.6942 | 0.3052 | 0.7834 | 0.4393 | 0.3402 | 0.7978 | 0.3881 |
| 7C | RF | 0.7521 | 0.3348 | 0.6294 | 0.4370 | 0.3210 | 0.7880 | 0.3790 |
| 7C | XGBoost | 0.7012 | 0.3072 | 0.7602 | 0.4376 | 0.3345 | 0.7989 | 0.3997 |
| 7C | LightGBM | 0.6928 | 0.3036 | 0.7805 | 0.4372 | 0.3371 | 0.8000 | 0.4014 |
| 7D | RF | 0.7423 | 0.3225 | 0.6228 | 0.4250 | 0.3051 | 0.7719 | 0.3616 |
| 7D | XGBoost | 0.6871 | 0.2950 | 0.7529 | 0.4239 | 0.3159 | 0.7814 | 0.3825 |
| 7D | LightGBM | 0.6808 | 0.2927 | 0.7682 | 0.4239 | 0.3181 | 0.7832 | 0.3826 |
| 7E | Stacking | 0.6998 | 0.3105 | 0.7892 | 0.4456 | 0.3491 | 0.8041 | 0.3966 |
| 7F | Weighted Blend | 0.7157 | 0.3204 | 0.7667 | 0.4519 | 0.3535 | 0.8049 | 0.4052 |

### 7F Blend Weights

Validation-selected:

- RF = 0.15
- XGBoost = 0.35
- LightGBM = 0.50

### Observations

- Adding all 51 LLM features to JIT + CodeBERT does not materially improve the main final-test metrics relative to 4F.
- 7F provides the principal full three-way fusion reference for Experiments 8 and 9.
- The results motivate testing whether the LLM representation can be compressed without losing predictive behavior.

---

# Experiment 8 — JIT + CodeBERT + Reduced LLM Representation

## Status: COMPLETED & FROZEN

### Objective

Test whether the 51-feature LLM representation can be reduced to the 14 confidence/margin features while preserving predictive performance.

### LLM Reduction

The seven semantic dimensions each contribute confidence and margin:

- intent confidence / margin
- change confidence / margin
- risk confidence / margin
- complexity confidence / margin
- scope confidence / margin
- test confidence / margin
- security confidence / margin

Thus:

**51 → 14 LLM features = 72.55% LLM dimensionality reduction**

Total model space:

**12 JIT + 384 CodeBERT + 14 LLM = 410 features**

### Validation-Selected Blend Weights

- RF = 0.15
- XGBoost = 0.40
- LightGBM = 0.45

### Final Test Results

| Model / Blend | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RF | 0.7650 | 0.3398 | 0.5698 | 0.4257 | 0.3050 | 0.7741 | 0.3637 |
| XGBoost | 0.7172 | 0.3214 | 0.7645 | 0.4526 | 0.3540 | 0.8026 | 0.4005 |
| LightGBM | 0.7006 | 0.3104 | 0.7849 | 0.4449 | 0.3475 | 0.8034 | 0.3993 |
| **Weighted Blend** | **0.7180** | **0.3223** | **0.7660** | **0.4537** | **0.3557** | **0.8046** | **0.4027** |

### Comparison with 7F

| Metric | 7F: 51 LLM | Exp. 8: 14 LLM |
| :--- | ---: | ---: |
| Total features | 447 | 410 |
| Accuracy | 0.7157 | 0.7180 |
| F1 | 0.4519 | 0.4537 |
| MCC | 0.3535 | 0.3557 |
| ROC-AUC | 0.8049 | 0.8046 |
| PR-AUC | 0.4052 | 0.4027 |

### Interpretation

The 14 confidence/margin features preserve performance close to the full 51-feature LLM representation while reducing LLM dimensionality by 72.55%. This supports a **compactness/redundancy** interpretation, not the stronger claim that fewer features universally improve prediction.

---

# Experiment 9 — Redundancy-Aware LLM Feature Selection

## Status: COMPLETED & FROZEN

### Objective

Test whether an alternative 14-feature LLM subset selected using explicit relevance and redundancy criteria improves on Experiment 8.

### Selection Procedure

1. Start with all 51 model-ready LLM features.
2. Fit categorical one-hot encoding on training data only.
3. Compute train-only Mutual Information with the buggy target.
4. Apply greedy redundancy filtering:
   - Pearson absolute correlation threshold: `|r| >= 0.90`
   - When two features are highly correlated, retain the higher-MI feature.
5. 51 features → 44 non-redundant candidates.
6. Select the top 14 remaining features by training Mutual Information.
7. Combine the selected 14 LLM features with 12 JIT + 384 CodeBERT.

### Selected LLM Features

1. `complexity_margin`
2. `scope_M`
3. `risk_confidence`
4. `scope_margin`
5. `complexity_M`
6. `test_M`
7. `test_confidence`
8. `security_margin`
9. `security_N`
10. `change_confidence`
11. `risk_M`
12. `change_OT`
13. `complexity_L`
14. `test_L`

### Feature Space

12 JIT + 384 CodeBERT + 14 selected LLM = **410 features**

### Validation-Selected Blend Weights

- RF = 0.10
- XGBoost = 0.80
- LightGBM = 0.10

### Final Test Results

| Model / Blend | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RF | 0.7686 | 0.3460 | 0.5770 | 0.4326 | 0.3139 | 0.7751 | 0.3589 |
| XGBoost | 0.7144 | 0.3190 | 0.7645 | 0.4502 | 0.3510 | 0.8022 | 0.3951 |
| LightGBM | 0.7007 | 0.3092 | 0.7762 | 0.4422 | 0.3428 | 0.8027 | 0.3989 |
| **Weighted Blend** | **0.7181** | **0.3219** | **0.7624** | **0.4526** | **0.3538** | **0.8031** | **0.3958** |

### Comparison with Experiment 8

| Metric | Exp. 8: Confidence/Margin | Exp. 9: MI + Correlation |
| :--- | ---: | ---: |
| LLM features | 14 | 14 |
| Total features | 410 | 410 |
| Accuracy | 0.7180 | 0.7181 |
| F1 | 0.4537 | 0.4526 |
| MCC | 0.3557 | 0.3538 |
| ROC-AUC | 0.8046 | 0.8031 |
| PR-AUC | 0.4027 | 0.3958 |

### Interpretation

Experiment 9 does **not** improve on Experiment 8. The train-only redundancy-aware procedure is therefore retained as a controlled negative finding.

The evidence supports the statement that the 51-feature LLM representation contains substantial redundancy and can be compressed substantially while preserving predictive behavior. It does **not** support the stronger claim that generic redundancy-aware feature selection improves prediction.

---

# Consolidated Comparison — Main Fusion Experiments

| Experiment | JIT | CodeBERT | LLM | Total Features | Accuracy | F1 | MCC | ROC-AUC | PR-AUC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **4F** | 12 | 384 | 0 | 396 | 0.7301 | 0.4548 | 0.3529 | 0.8054 | 0.4043 |
| **7F** | 12 | 384 | 51 | 447 | 0.7157 | 0.4519 | 0.3535 | 0.8049 | 0.4052 |
| **8** | 12 | 384 | 14 confidence/margin | 410 | 0.7180 | 0.4537 | 0.3557 | 0.8046 | 0.4027 |
| **9** | 12 | 384 | 14 selected | 410 | 0.7181 | 0.4526 | 0.3538 | 0.8031 | 0.3958 |

## Main Findings

1. **4F establishes the JIT + CodeBERT fusion reference.**
2. **7F shows that adding all 51 LLM features does not materially improve the main predictive metrics over 4F.**
3. **Experiment 8 reduces the LLM representation from 51 to 14 features (72.55% reduction) while preserving comparable predictive performance.**
4. **Experiment 9 demonstrates that generic train-only Mutual Information + correlation-based redundancy selection does not provide an additional improvement over the simpler confidence/margin representation.**
5. The strongest defensible interpretation is therefore **substantial LLM feature redundancy with successful compact representation**, rather than a claim that generic feature selection improves prediction.
6. All feature-selection, PCA, encoding, imputation, and ensemble-weight decisions are based only on training/validation information; the final test set remains untouched until evaluation.

---

# Experiment Status Summary

| Experiment | Description | Status |
| :--- | :--- | :--- |
| 1 | JIT-only | **FROZEN** |
| 2 | CodeBERT-only | **FROZEN** |
| 3 | LLM-only | **FROZEN** |
| 4A–4F | JIT + CodeBERT fusion variants | **FROZEN** |
| 5 | JIT + LLM | **FROZEN** |
| 6 | CodeBERT + LLM | **FROZEN** |
| 7A–7F | JIT + CodeBERT + full LLM | **FROZEN** |
| 8 | JIT + CodeBERT + 14 confidence/margin LLM features | **FROZEN** |
| 9 | JIT + CodeBERT + redundancy-aware 14-feature LLM selection | **FROZEN** |

## Recommended Paper/Thesis Wording

> Reducing the LLM representation from 51 to 14 confidence- and margin-based features reduced the LLM dimensionality by 72.5% while preserving comparable predictive performance in the full JIT + CodeBERT fusion setting. A subsequent train-only mutual-information and correlation-based redundancy-selection procedure did not further improve performance, suggesting that compactness can be achieved without requiring generic redundancy-aware feature selection.