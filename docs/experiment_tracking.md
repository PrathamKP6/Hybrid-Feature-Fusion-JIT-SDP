# Experiment Tracking

# Experiment 1 — JIT Metrics Only

## Objective

Evaluate traditional JIT metrics for defect prediction.

## Features Used

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

## Models

* Random Forest
* XGBoost

## Dataset

* Filtered years: 2018–2026
* Stratified sampling:
  Language × Bug Label
* Sample size: ~10,000 commits

## Temporal Split

* Train: First 80%
* Test: Last 20%

## Results

| Model | Accuracy | F1     | ROC-AUC | MCC    |
| ----- | -------- | ------ | ------- | ------ |
| RF    | 0.7075   | 0.4167 | 0.7578  | 0.2400 |
| XGB   | 0.737    | 0.6075 | 0.8029  | 0.4178 |

## Observations:

- XGBoost consistently outperformed Random Forest across all evaluation metrics.

- The largest improvement was observed in F1-score (0.4167 to 0.6075), indicating superior handling of buggy commits.

- ROC-AUC of 0.8029 demonstrates that traditional JIT metrics possess strong predictive capability for defect prediction.

- MCC improved substantially from 0.2400 to 0.4178, suggesting better learning of the minority class.

- Traditional JIT metrics provide a strong baseline for future experiments involving semantic and reasoning-based features.

- The experiment confirms that code churn and developer history features contain useful defect-related information, while leaving room for further improvement through semantic feature integration.

---

# Experiment 2 — CodeBERT Only

## Objective

Evaluate semantic code representations.

## Features Used

* PCA-reduced CodeBERT features (384 dimensions)

## Models

* Random Forest
* XGBoost

## Results

TBD
