"""
Generate publication and reproducibility documentation for the final dataset.

The script validates only the publication CSV header and writes DATASET.md.
It never loads or modifies the dataset contents.
"""

from pathlib import Path
import logging
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIR = ROOT / "data" / "public_release"
DATASET = PUBLIC_DIR / "final_multilingual_jit_codebert_llm_59996_768d.csv"
OUTPUT = PUBLIC_DIR / "DATASET.md"


logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


EXPECTED_COLUMNS = [
	"commit_id", "project", "buggy", "fix", "year", "author_date",
	"la", "ld", "nf", "nd", "ns", "ent", "ndev", "age", "nuc",
	"aexp", "arexp", "asexp", "message", "diff", "embedding", "intent",
	"intent_confidence", "intent_margin", "change", "change_confidence",
	"change_margin", "risk", "risk_confidence", "risk_margin", "complexity",
	"complexity_confidence", "complexity_margin", "scope", "scope_confidence",
	"scope_margin", "test", "test_confidence", "test_margin", "security",
	"security_confidence", "security_margin", "model", "schema_version",
]


DATASET_MARKDOWN = r'''# Final Multilingual JIT, CodeBERT, and LLM Dataset

## 1. Overview

This release contains the final publication dataset for reproducible just-in-time software defect prediction (JIT-SDP) experiments. It contains **59,996 commits** and **44 stored CSV columns**. The CSV is the source artifact; this document describes its stored fields, model representations, temporal protocol, and integrity checks.

## 2. Dataset Composition

- Total commits: 59,996
- Unique commit IDs: 59,996
- Projects: 19
- Languages: Java, C++, and Python
- Approximate composition: Java 36,000 (60%), C++ 12,000 (20%), Python 11,996 (20%)
- The CSV contains `project`, but no separate `language` column.

## 3. Projects

The dataset covers 19 source projects. Project identity is stored in `project`; language is documented as dataset-level composition and is not stored in a separate CSV field.

## 4. Dataset Columns

The stored columns, in order, are:

| # | Column | Role |
|---:|---|---|
| 1 | `commit_id` | Commit identifier |
| 2 | `project` | Project metadata |
| 3 | `buggy` | Target label |
| 4 | `fix` | Source metadata |
| 5 | `year` | Temporal metadata |
| 6 | `author_date` | Temporal metadata |
| 7 | `la` | JIT metric: lines added |
| 8 | `ld` | JIT metric: lines deleted |
| 9 | `nf` | JIT metric: modified files |
| 10 | `nd` | JIT metric: modified directories |
| 11 | `ns` | JIT metric: modified subsystems |
| 12 | `ent` | JIT metric: change entropy |
| 13 | `ndev` | JIT metric: number of developers |
| 14 | `age` | JIT metric: change age |
| 15 | `nuc` | JIT metric: unique changes |
| 16 | `aexp` | JIT metric: developer experience |
| 17 | `arexp` | JIT metric: recent developer experience |
| 18 | `asexp` | JIT metric: subsystem experience |
| 19 | `message` | Commit message |
| 20 | `diff` | Code diff |
| 21 | `embedding` | One serialized 768-dimensional CodeBERT vector |
| 22 | `intent` | LLM change intent |
| 23 | `intent_confidence` | Intent confidence |
| 24 | `intent_margin` | Intent margin |
| 25 | `change` | LLM change type |
| 26 | `change_confidence` | Change confidence |
| 27 | `change_margin` | Change margin |
| 28 | `risk` | LLM change risk |
| 29 | `risk_confidence` | Risk confidence |
| 30 | `risk_margin` | Risk margin |
| 31 | `complexity` | LLM change complexity |
| 32 | `complexity_confidence` | Complexity confidence |
| 33 | `complexity_margin` | Complexity margin |
| 34 | `scope` | LLM change scope |
| 35 | `scope_confidence` | Scope confidence |
| 36 | `scope_margin` | Scope margin |
| 37 | `test` | LLM testing characteristic |
| 38 | `test_confidence` | Test confidence |
| 39 | `test_margin` | Test margin |
| 40 | `security` | LLM security characteristic |
| 41 | `security_confidence` | Security confidence |
| 42 | `security_margin` | Security margin |
| 43 | `model` | LLM provenance metadata |
| 44 | `schema_version` | LLM schema provenance metadata |

The complete terminology and descriptions are also provided in `FEATURE_DICTIONARY.md`.

## 5. Target Label

The target column is `buggy`. The canonical normalized SZZ-based label is authoritative:

- `0`: non-buggy, 44,626 commits
- `1`: buggy, 15,370 commits

The original LLM datasets contained their own `buggy` column. It was verified against the canonical label and had zero actual mismatches after normalization, so the duplicate LLM column was not retained in the final publication CSV.

## 6. Traditional JIT Features

The final JIT representation contains exactly 12 features: `la`, `ld`, `nf`, `ns`, `nd`, `ent`, `ndev`, `age`, `nuc`, `aexp`, `arexp`, and `asexp`. The `fix` column is retained as source metadata and is not part of this 12-feature representation.

## 7. Commit Message and Code Diff

`message` stores the original commit message and `diff` stores the textual code change representation. They are retained source fields and are not counted as traditional JIT features.

## 8. CodeBERT Representation

`embedding` is one CSV column containing a serialized 768-dimensional numerical CodeBERT vector. It must not be counted as one model feature. The reported fusion experiments use a reduced representation after train-only PCA.

## 9. PCA Reduction

PCA was fitted **only on training data**. Validation and test rows were transformed with that same training-fitted PCA model. The selected representation has 384 dimensions and explains approximately 99.4280% of the variance. PCA was not fitted on the full dataset.

## 10. LLM Semantic Representation

The LLM representation has seven categorical semantic dimensions: `intent`, `change`, `risk`, `complexity`, `scope`, `test`, and `security`. Each dimension has one categorical output, one confidence value, and one margin value. Thus, the raw semantic representation has 7 categorical columns, 7 confidence columns, and 7 margin columns: 21 raw LLM semantic columns.

## 11. LLM Model-Ready Representation

During preprocessing, the seven categorical dimensions are one-hot encoded into 37 categorical features. The 14 confidence and margin columns remain numerical features. The full model-ready LLM representation therefore has **51 features**. This is distinct from the 44 stored CSV columns.

## 12. Compact LLM Representation

The compact LLM representation uses the 7 confidence features and 7 margin features only, for a total of **14 numerical features**. Categorical outputs are omitted from this compact representation.

## 13. Temporal Evaluation Protocol

The split is project-wise chronological 70/15/15. Within each project, commits are ordered chronologically: the oldest 70% go to training, the next 15% to validation, and the newest 15% to test. There is no random train/test split, no random shuffling for the temporal partition, and no resampling.

Exact final sizes are:

| Split | Commits |
|---|---:|
| Train | 41,998 |
| Validation | 8,998 |
| Test | 9,000 |
| **Total** | **59,996** |

## 14. Split Reproducibility

Exact commit IDs are supplied in `split_ids/train_ids.txt`, `split_ids/validation_ids.txt`, and `split_ids/test_ids.txt`. The corresponding summary is in `split_ids/split_summary.txt`. Researchers should use these IDs rather than reconstructing the partition with a random split.

## 15. Leakage-Controlled Preprocessing

All preprocessing and model-selection operations respect the temporal split:

- PCA was fitted on training data only.
- One-hot encoding was fitted on training data only.
- Numerical imputation, where applicable, was fitted on training data only.
- Feature-selection procedures were fitted on training data only.
- Ensemble weights were selected using validation data only.
- Test data were reserved for final evaluation.
- No test-set information was used for preprocessing or model-selection decisions.

## 16. Model Features by Experiment

| Representation | Features |
|---|---:|
| JIT only | 12 |
| CodeBERT only | 384 |
| LLM only | 51 |
| JIT + CodeBERT | 396 |
| JIT + LLM | 63 |
| CodeBERT + LLM | 435 |
| JIT + CodeBERT + LLM | 447 |
| Experiment 8 compact LLM | 410 |
| Experiment 9 selected LLM | 410 |

The full three-way representation is 12 JIT + 384 PCA CodeBERT + 51 LLM = **447 model features**. The 768-dimensional embedding is stored in one serialized column, while the 384-dimensional representation is used in the reported fusion experiments after train-only PCA.

## 17. LLM Provenance

`model` identifies the LLM used to generate semantic features and `schema_version` identifies the semantic-feature schema. Both are provenance metadata and are not predictive features.

## 18. Information Not Intended as Model Features

`commit_id`, `project`, `fix`, `year`, `author_date`, `message`, and `diff` are stored for identification, verification, temporal ordering, source context, or reproducibility. They are not part of the listed model representations. `model` and `schema_version` are also metadata, not predictive features. The `buggy` column is the target and must not be used as an input feature.

## 19. Reproducibility Files

The `data/public_release/` directory contains:

- `final_multilingual_jit_codebert_llm_59996_768d.csv`
- `final_multilingual_jit_codebert_llm_59996_768d.zip`
- `SHA256SUMS.txt`
- `AUDIT_REPORT.txt`
- `FEATURE_DICTIONARY.csv`
- `FEATURE_DICTIONARY.md`
- `DATASET.md`
- `split_ids/train_ids.txt`
- `split_ids/validation_ids.txt`
- `split_ids/test_ids.txt`
- `split_ids/split_summary.txt`

## 20. Dataset Integrity

The publication audit passed and verified 59,996 rows, 59,996 unique commit IDs, binary `buggy` labels, 12 JIT features, 768-dimensional embeddings, 19 projects, no missing values, the train/validation/test counts, no split overlap, complete split coverage, and ZIP integrity.

## 21. Recommended Usage

Use the canonical `buggy` column as the target, preserve the supplied temporal split, and fit every learned preprocessing step on training data only. For CodeBERT fusion experiments, parse the serialized `embedding` vector and apply the training-fitted PCA transformation to validation and test data. Do not use identifiers, target values, or provenance metadata as predictive inputs.

## 22. Relationship to the Research Repository

This publication directory is the reproducibility-facing release artifact of the research repository. The repository contains the mining, preprocessing, embedding, experiment, and analysis code used to produce and evaluate the representations described here.

## 23. Important Reproducibility Note

The 44 stored CSV columns are a storage schema, not a model-feature count. In particular, one serialized 768-dimensional embedding column becomes 384 dimensions after train-only PCA in the reported fusion experiments, and the 21 raw LLM semantic columns become 51 model-ready features after one-hot encoding.

## 24. Summary

This dataset combines 12 traditional JIT metrics, a serialized 768-dimensional CodeBERT representation, and seven LLM semantic dimensions with confidence and margin values. It uses authoritative canonical SZZ labels and fixed project-wise chronological splits totaling 59,996 commits. The documented train-only preprocessing protocol supports leakage-controlled comparison of the listed model representations.

## 25. Files for Researchers

Start with the CSV, `FEATURE_DICTIONARY.md`, `AUDIT_REPORT.txt`, and the three split-ID files. Use `SHA256SUMS.txt` to verify the downloaded artifacts before beginning experiments.
'''


def validate_schema() -> None:
	"""Validate the publication CSV header without loading its rows."""
	if not DATASET.exists():
		raise FileNotFoundError(f"Input publication dataset not found: {DATASET}")

	header = pd.read_csv(DATASET, nrows=0, low_memory=False).columns.tolist()
	if header != EXPECTED_COLUMNS:
		missing = [column for column in EXPECTED_COLUMNS if column not in header]
		unexpected = [column for column in header if column not in EXPECTED_COLUMNS]
		raise ValueError(
			"Publication dataset schema differs from the expected 44-column "
			f"schema. Expected {len(EXPECTED_COLUMNS)} columns, found {len(header)}. "
			f"Missing: {missing or 'none'}. Unexpected: {unexpected or 'none'}."
		)

	logger.info("PASS: publication schema verified (%d columns).", len(header))


def main() -> int:
	logger.info("Generating publication dataset documentation")
	validate_schema()
	OUTPUT.write_text(DATASET_MARKDOWN, encoding="utf-8")
	logger.info("Wrote publication documentation: %s", OUTPUT)
	print(f"SUCCESS: generated {OUTPUT}")
	return 0


if __name__ == "__main__":
	try:
		raise SystemExit(main())
	except Exception as exc:
		logger.error("FAILED: %s", exc)
		print(f"ERROR: {exc}", file=sys.stderr)
		raise SystemExit(1)
