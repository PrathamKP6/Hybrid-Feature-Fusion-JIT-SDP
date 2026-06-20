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

| Model | Accuracy | F1     | ROC-AUC | MCC    |
| ----- | -------- | ------ | ------- | ------ |
| RF    | 0.698    | 0.0591 | 0.7157  | 0.0514 |
| XGB   | 0.7095   | 0.1315 | 0.7420  | 0.1418 |

## Observations:

## Observations

- XGBoost outperformed Random Forest across all evaluation metrics, indicating that boosting methods are more effective in utilizing semantic CodeBERT representations for defect prediction.

- Despite achieving moderate ROC-AUC values (0.7157 for RF and 0.7420 for XGB), both models produced very low F1-scores and MCC values, suggesting difficulty in correctly identifying bug-inducing commits.

- Compared to Experiment 1 (JIT Metrics Only), CodeBERT features alone resulted in substantially lower predictive performance. The best CodeBERT model (XGBoost) achieved:

   * F1-score: 0.1315 vs 0.6075
   * ROC-AUC: 0.7420 vs 0.8029
   * MCC: 0.1418 vs 0.4178

- The large performance gap indicates that semantic embeddings alone are insufficient for accurate Just-in-Time defect prediction within the current experimental setup.

- Although CodeBERT captures semantic characteristics of code changes, it does not explicitly encode important historical and process-related information such as code churn, developer experience, subsystem history, and change entropy, which are available in traditional JIT metrics.

- The relatively low MCC values suggest limited capability in correctly handling the minority buggy class, despite the semantic information contained in the embeddings.

- The results imply that semantic representations may serve better as complementary features rather than standalone predictors of software defects.

- The experiment demonstrates that contextual software engineering information remains critical for defect prediction and motivates feature fusion experiments combining JIT metrics with CodeBERT embeddings.

---

## Key Findings

* Traditional JIT metrics significantly outperformed standalone CodeBERT semantic features.
* Semantic embeddings alone were unable to capture defect-inducing patterns as effectively as handcrafted software engineering metrics.
* CodeBERT representations may still provide useful complementary information when combined with JIT metrics.
* These results strongly justify the need for feature fusion experiments (Experiments 4 and 7) to investigate whether semantic information can enhance traditional defect prediction models rather than replace them.

---

