# Feature Descriptions

# Traditional JIT Metrics

| Feature | Description                               |
| ------- | ----------------------------------------- |
| LA      | Lines Added                               |
| LD      | Lines Deleted                             |
| NF      | Number of Files Modified                  |
| NS      | Number of Subsystems Modified             |
| ND      | Number of Directories Modified            |
| Entropy | Distribution of code changes across files |
| NDEV    | Number of Developers                      |
| AGE     | Average time since last modification      |
| NUC     | Number of Unique Changes                  |
| AEXP    | Author Experience                         |
| AREXP   | Recent Author Experience                  |
| ASEXP   | Subsystem Experience                      |

---

# CodeBERT Features

## PCA Features

| Feature | Description       |
| ------- | ----------------- |
| pca_1   | PCA Component 1   |
| pca_2   | PCA Component 2   |
| ...     | ...               |
| pca_384 | PCA Component 384 |

These features are derived from PCA-reduced CodeBERT embeddings.

---

# LLM Reasoning Features

## Intent Features

| Feature                 | Description                    |
| ----------------------- | ------------------------------ |
| intent_bug_fix          | Indicates bug fixing intent    |
| intent_refactoring      | Indicates refactoring changes  |
| intent_feature_addition | Indicates new feature addition |

---

## Risk Features

| Feature    | Description           |
| ---------- | --------------------- |
| risk_level | Estimated defect risk |

Encoding:

* low = 0
* medium = 1
* high = 2
* critical = 3

---

## Complexity Features

| Feature              | Description                         |
| -------------------- | ----------------------------------- |
| reasoning_complexity | Estimated implementation complexity |

---

## Scope Features

| Feature      | Description                  |
| ------------ | ---------------------------- |
| change_scope | Estimated modification scope |

---

## Confidence Features

| Feature          | Description          |
| ---------------- | -------------------- |
| confidence_score | LLM confidence score |
