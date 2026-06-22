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

# Experiment 4a — JIT + CodeBERT Fusion

## Objective

Evaluate whether combining traditional JIT metrics with PCA-reduced CodeBERT features improves defect prediction.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | MCC |
| ----- | -------- | --------- | ------ | --- | ------- | ------ | ----- |
| Random Forest | 0.6965 | 0.2000 | 0.0033 | 0.0065 | 0.6869 | 0.4042 | -0.0155 |
| XGBoost | 0.6915 | 0.4708 | 0.2146 | 0.2949 | 0.6867 | 0.4309 | 0.1480 |

## Observations:

- XGBoost outperformed Random Forest, but the fusion model still underperformed the JIT-only baseline.
- The best XGBoost model produced low recall and moderate precision, suggesting the fused feature set added noise rather than signal.
- The ROC-AUC of 0.6867 is notably below the JIT-only performance, indicating that naive feature concatenation with CodeBERT PCA features did not improve predictive quality.
- This result highlights the importance of careful feature selection and model-level fusion when combining semantic and process features.

---

# Experiment 4b — Feature Selection for Fusion

## Objective

Identify whether selecting the most important PCA features improves JIT + CodeBERT fusion performance.

## Results

* Best model: XGBoost
* Best selection method: RF importance
* Best top-k: 25
* ROC-AUC: 0.7919
* PR-AUC: 0.6052
* F1: 0.5841
* MCC: 0.3987

## Observations:

- Selecting the top 25 PCA features improved fusion performance relative to unselected PCA fusion.
- However, the best fusion result still remained below the JIT-only ROC-AUC baseline.
- The fact that RF-based feature importance yielded the best subset suggests that model-guided selection is more effective than raw PCA alone.
- This experiment indicates that semantic feature selection can reduce noise, but feature-level fusion still has limited benefit without stronger alignment to the defect signal.

---

# Experiment 4c — Top-K PCA Selection

## Objective

Evaluate how many top PCA dimensions are needed to maximize hybrid fusion performance.

## Results

* Best k: 15
* ROC-AUC: 0.7913
* PR-AUC: 0.5933
* F1: 0.5860
* MCC: 0.4008
* Precision: 0.5690
* Recall: 0.6040
* Accuracy: 0.7435

## Observations:

- The top 15 PCA features produced the best hybrid fusion performance in this sweep.
- Even with aggressive dimensionality reduction, the best top-k fusion still did not exceed the strong JIT-only baseline.
- This suggests that a compact semantic representation can preserve much of the useful signal, but additional fusion complexity is required to exceed traditional metrics.
- The relatively balanced precision and recall at k=15 indicate a useful tradeoff between model sensitivity and specificity.

---

# Experiment 4d — Late Fusion / Stacking Ensemble

## Objective

Evaluate whether model-level fusion of JIT and semantic predictions can recover complementary information.

## Results (default thresholds)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | MCC |
| ----- | -------- | --------- | ------ | --- | ------- | ------ | ----- |
| JIT Only | 0.737 | 0.5507 | 0.6772 | 0.6075 | 0.8029 | 0.6223 | 0.4178 |
| PCA Only | 0.651 | 0.4159 | 0.3993 | 0.4075 | 0.6533 | 0.3859 | 0.1603 |
| Early Fusion | 0.6915 | 0.4708 | 0.2146 | 0.2949 | 0.6867 | 0.4309 | 0.1480 |
| Top-25 Fusion | 0.735 | 0.5583 | 0.5657 | 0.5620 | 0.7817 | 0.5814 | 0.3720 |
| Stacking Ensemble | 0.707 | 0.7027 | 0.0433 | 0.0815 | 0.8040 | 0.6127 | 0.1204 |

## Threshold tuning

* Best threshold: 0.1
* Precision: 0.5027
* Recall: 0.7787
* F1: 0.6110
* MCC: 0.4115

## Observations:

- The stacking ensemble achieved the highest ROC-AUC among all tested configurations.
- Default classification threshold produced very low recall and poor F1, demonstrating the need for threshold tuning.
- After tuning to threshold 0.1, the stacking model slightly improved F1 over the JIT-only baseline and matched its ranking performance more closely.
- The JIT-only baseline remained stronger on MCC and precision, indicating that ensemble fusion added recall at the cost of some precision.
- This experiment shows that late fusion is a viable way to leverage complementary semantic predictions, but threshold calibration is essential.

---

# Experiment 4e — Feature Group Ablation

## Objective

Compare JIT-only, PCA-only, and JIT+CodeBERT fusion to determine which feature group drives the best performance.

## Results

* Best model: JIT-only
* JIT-only ROC-AUC: 0.8029
* PCA-only ROC-AUC: 0.6819
* Fusion ROC-AUC: 0.6869
* JIT-only PR-AUC: 0.6223
* Fusion PR-AUC: 0.4309
* Fusion gain over JIT: -0.1160
* Fusion gain over PCA: +0.0050

## Observations:

- JIT-only remained the strongest configuration, outperforming both PCA-only and JIT+CodeBERT fusion.
- The fusion model degraded performance relative to JIT-only, showing that the added semantic features did not improve the baseline in this ablation setting.
- Fusion offered a small improvement over PCA-only, indicating that JIT features are the dominant signal.
- These results reinforce the conclusion that semantic CodeBERT features are most useful when carefully combined with process metrics and calibrated at the model level.

---

