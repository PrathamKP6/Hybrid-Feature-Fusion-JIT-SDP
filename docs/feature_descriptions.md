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

Intent: 
| Code | Meaning                                                          |
| ---- | ---------------------------------------------------------------- |
| `BF` | Bug Fix — corrects an existing defect                            |
| `FT` | Feature — adds new functionality                                 |
| `RF` | Refactoring — restructures code without intended behavior change |
| `DC` | Documentation — documentation/comments/readme changes            |
| `TS` | Testing — adds/modifies/removes tests or test infrastructure     |
| `BL` | Build/Dependency — build system, dependencies, packaging         |
| `PF` | Performance — improves performance/resource efficiency           |
| `SC` | Security — specifically addresses a security issue/control       |
| `OT` | Other — intent doesn't fit the above                             |

Change Type:
| Code | Meaning                                                             |
| ---- | ------------------------------------------------------------------- |
| `LG` | Logic — changes program logic/control flow                          |
| `AP` | API — changes interfaces, endpoints, public methods/contracts       |
| `DA` | Data — changes data structures, schemas, serialization, persistence |
| `CF` | Configuration — configuration/environment/settings changes          |
| `UI` | User Interface — UI/frontend/presentation changes                   |
| `DP` | Dependency — adds, removes, upgrades, or changes dependencies       |
| `TS` | Test — modifies test code/infrastructure                            |
| `DC` | Documentation — documentation/comments                              |
| `BL` | Build/Release — build, CI/CD, packaging, release                    |
| `OT` | Other                                                               |


Risk level

| Code | Meaning                                                                 |
| ---- | ----------------------------------------------------------------------- |
| `L`  | Low — localized/simple change with limited failure surface              |
| `M`  | Medium — meaningful behavioral or integration impact                    |
| `H`  | High — broad/complex/critical behavior or substantial failure potential |


Complexity

| Code | Meaning                                                                                     |
| ---- | ------------------------------------------------------------------------------------------- |
| `L`  | Low — simple/localized modification                                                         |
| `M`  | Medium — multiple interacting changes or moderate logic                                     |
| `H`  | High — complex logic, architecture, concurrency, algorithms, or many interacting components |


Scope: 

| Code | Meaning                                                   |
| ---- | --------------------------------------------------------- |
| `L`  | Local — one small component/file/function                 |
| `F`  | Focused — a small set of closely related files/components |
| `M`  | Multi-component — multiple components/modules             |
| `C`  | Cross-cutting — affects many parts/layers of the system   |

Test Impact:

| Code | Meaning                                                                                                |
| ---- | ------------------------------------------------------------------------------------------------------ |
| `N`  | None/minimal — little or no meaningful testing impact                                                  |
| `L`  | Low — existing tests likely sufficient; small test adjustment                                          |
| `M`  | Medium — new/modified tests or meaningful regression coverage needed                                   |
| `H`  | High — substantial new testing, integration testing, regression testing, or difficult-to-test behavior |


Security Risk:

| Code | Meaning                                                                                                                                |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `N`  | None — no meaningful security relevance                                                                                                |
| `L`  | Low — indirect/minor security relevance                                                                                                |
| `M`  | Medium — affects authentication, authorization, input validation, data protection, etc.                                                |
| `H`  | High — directly changes critical security controls, vulnerabilities, cryptography, privilege boundaries, sensitive-data handling, etc. |


Confidence 0-100
Confidence should represent confidence in the overall 7 categorical classifications, not confidence in whether the commit is defective.

means approximately:

"I am 91% confident that my classification of the seven semantic features is correct."

It does not mean:

"There is a 91% probability this commit contains a defect."