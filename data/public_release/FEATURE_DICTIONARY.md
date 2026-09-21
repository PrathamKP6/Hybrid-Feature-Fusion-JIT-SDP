# Publication Dataset Feature Dictionary

The final publication dataset contains **59,996 commits and 44 stored columns**.

The 44-column storage schema should not be confused with the number of model-ready features. The LLM representation contains 21 raw semantic columns (7 categorical + 14 confidence/margin columns). After one-hot encoding the seven categorical dimensions, these become 37 categorical features plus 14 numerical features, for a total of **51 model-ready LLM features**.

## Dataset-Level Feature Summary

| Representation | Stored columns | Model-ready dimensions |
|---|---:|---:|
| Metadata / identifiers | 2 | 0 |
| Target | 1 | 1 |
| Other source metadata | 3 | 0 |
| JIT metrics | 12 | 12 |
| Commit message | 1 | 0 |
| Code diff | 1 | 0 |
| CodeBERT embedding | 1 serialized vector | 768 / 384* |
| LLM categorical outputs | 7 | 37 after one-hot encoding |
| LLM confidence/margin | 14 | 14 |
| LLM provenance | 2 | 0 |

*The fusion experiments using CodeBERT apply train-only PCA from 768 to 384 dimensions.

## Complete Feature Dictionary

| # | Column | Feature | Category | Description |
|---:|---|---|---|---|
| 1 | `commit_id` | Commit identifier | Metadata / identifier | Unique identifier of the source commit. Used for joining, verification, and split identification; never used as a model feature. |
| 2 | `project` | Project | Metadata | Source repository/project associated with the commit. |
| 3 | `buggy` | Buggy label | Target | Binary defect label derived from the canonical SZZ-based labeling pipeline. 0 = non-buggy, 1 = buggy. |
| 4 | `fix` | Fix indicator | Metadata / source field | Original commit-level fix indicator retained from the canonical dataset. |
| 5 | `year` | Commit year | Metadata | Year associated with the commit author date. |
| 6 | `author_date` | Author date | Temporal metadata | Commit author timestamp used as part of chronological ordering. |
| 7 | `la` | Lines added | JIT metric | Number of lines added by the commit. |
| 8 | `ld` | Lines deleted | JIT metric | Number of lines deleted by the commit. |
| 9 | `nf` | Number of modified files | JIT metric | Number of files modified by the commit. |
| 10 | `nd` | Number of modified directories | JIT metric | Number of directories affected by the commit. |
| 11 | `ns` | Number of modified subsystems | JIT metric | Number of subsystems affected by the commit. |
| 12 | `ent` | Change entropy | JIT metric | Entropy-based measure of how changes are distributed across the modified files. |
| 13 | `ndev` | Number of developers | JIT metric | Number of developers associated with the relevant project/history used by the JIT metric computation. |
| 14 | `age` | Change age | JIT metric | Age-related temporal metric associated with the modified code. |
| 15 | `nuc` | Number of unique changes | JIT metric | Historical change-count metric associated with the modified code. |
| 16 | `aexp` | Developer experience | JIT metric | Developer experience measure used by the JIT defect prediction model. |
| 17 | `arexp` | Recent developer experience | JIT metric | Recent developer experience measure used by the JIT model. |
| 18 | `asexp` | Subsystem experience | JIT metric | Developer subsystem-experience measure used by the JIT model. |
| 19 | `message` | Commit message | Source text | Original commit message associated with the change. |
| 20 | `diff` | Code diff | Source text | Original textual representation of the code changes introduced by the commit. |
| 21 | `embedding` | CodeBERT embedding | Semantic representation | 768-dimensional CodeBERT representation of the commit/change content. Stored as one serialized vector column. For the experiments using PCA, PCA was fitted on training data only and validation/test data were transformed using the same fitted model. |
| 22 | `intent` | Change intent | LLM categorical semantic feature | LLM-generated semantic classification describing the primary intent of the change. |
| 23 | `intent_confidence` | Intent confidence | LLM numerical semantic feature | LLM confidence associated with the predicted intent category. |
| 24 | `intent_margin` | Intent margin | LLM numerical semantic feature | Margin associated with the LLM intent prediction. |
| 25 | `change` | Change type | LLM categorical semantic feature | LLM-generated classification describing the type of change. |
| 26 | `change_confidence` | Change confidence | LLM numerical semantic feature | LLM confidence associated with the predicted change category. |
| 27 | `change_margin` | Change margin | LLM numerical semantic feature | Margin associated with the LLM change prediction. |
| 28 | `risk` | Change risk | LLM categorical semantic feature | LLM-generated categorical assessment of change risk. |
| 29 | `risk_confidence` | Risk confidence | LLM numerical semantic feature | LLM confidence associated with the predicted risk category. |
| 30 | `risk_margin` | Risk margin | LLM numerical semantic feature | Margin associated with the LLM risk prediction. |
| 31 | `complexity` | Change complexity | LLM categorical semantic feature | LLM-generated categorical assessment of change complexity. |
| 32 | `complexity_confidence` | Complexity confidence | LLM numerical semantic feature | LLM confidence associated with the predicted complexity category. |
| 33 | `complexity_margin` | Complexity margin | LLM numerical semantic feature | Margin associated with the LLM complexity prediction. |
| 34 | `scope` | Change scope | LLM categorical semantic feature | LLM-generated categorical assessment of the scope of the change. |
| 35 | `scope_confidence` | Scope confidence | LLM numerical semantic feature | LLM confidence associated with the predicted scope category. |
| 36 | `scope_margin` | Scope margin | LLM numerical semantic feature | Margin associated with the LLM scope prediction. |
| 37 | `test` | Testing characteristic | LLM categorical semantic feature | LLM-generated classification describing the testing characteristic of the change. |
| 38 | `test_confidence` | Test confidence | LLM numerical semantic feature | LLM confidence associated with the predicted testing category. |
| 39 | `test_margin` | Test margin | LLM numerical semantic feature | Margin associated with the LLM testing prediction. |
| 40 | `security` | Security characteristic | LLM categorical semantic feature | LLM-generated classification describing the security characteristic of the change. |
| 41 | `security_confidence` | Security confidence | LLM numerical semantic feature | LLM confidence associated with the predicted security category. |
| 42 | `security_margin` | Security margin | LLM numerical semantic feature | Margin associated with the LLM security prediction. |
| 43 | `model` | LLM model | Provenance metadata | Identifier of the LLM model used to generate the semantic features. |
| 44 | `schema_version` | LLM feature schema version | Provenance metadata | Version identifier for the LLM semantic-feature schema. |

## LLM Representation

The stored LLM semantic representation contains seven categorical dimensions:

1. Intent  
2. Change  
3. Risk  
4. Complexity  
5. Scope  
6. Test  
7. Security

Each categorical dimension has an associated confidence and margin, giving 14 numerical LLM features.

**Raw LLM semantic columns:** 7 categorical + 14 numerical = 21 columns.

**Model-ready LLM representation:** 37 one-hot categorical features + 14 numerical confidence/margin features = **51 features**.

The `model` and `schema_version` columns are provenance metadata and are not used as predictive features.

## Model Feature Counts

| Experiment representation | Feature dimensions |
|---|---:|
| JIT only | 12 |
| CodeBERT only | 384 |
| LLM only | 51 |
| JIT + CodeBERT | 396 |
| JIT + LLM | 63 |
| CodeBERT + LLM | 435 |
| JIT + CodeBERT + LLM | 447 |
| Exp. 8 compact LLM | 410 |
| Exp. 9 selected LLM | 410 |
