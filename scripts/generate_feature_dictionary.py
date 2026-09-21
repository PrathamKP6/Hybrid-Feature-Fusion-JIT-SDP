"""
generate_feature_dictionary.py

Generate the publication feature dictionary for the final
44-column reproducibility dataset.

Input:
    data/public_release/
        final_multilingual_jit_codebert_llm_59996_768d.csv

Output:
    data/public_release/
        FEATURE_DICTIONARY.csv
        FEATURE_DICTIONARY.md
"""

from pathlib import Path
import logging
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PUBLIC_DIR = ROOT / "data" / "public_release"

DATASET = (
    PUBLIC_DIR
    / "final_multilingual_jit_codebert_llm_59996_768d.csv"
)

CSV_OUTPUT = PUBLIC_DIR / "FEATURE_DICTIONARY.csv"
MD_OUTPUT = PUBLIC_DIR / "FEATURE_DICTIONARY.md"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Expected publication schema
# ----------------------------------------------------------------------

EXPECTED_COLUMNS = [
    "commit_id",
    "project",
    "buggy",
    "fix",
    "year",
    "author_date",
    "la",
    "ld",
    "nf",
    "nd",
    "ns",
    "ent",
    "ndev",
    "age",
    "nuc",
    "aexp",
    "arexp",
    "asexp",
    "message",
    "diff",
    "embedding",
    "intent",
    "intent_confidence",
    "intent_margin",
    "change",
    "change_confidence",
    "change_margin",
    "risk",
    "risk_confidence",
    "risk_margin",
    "complexity",
    "complexity_confidence",
    "complexity_margin",
    "scope",
    "scope_confidence",
    "scope_margin",
    "test",
    "test_confidence",
    "test_margin",
    "security",
    "security_confidence",
    "security_margin",
    "model",
    "schema_version",
]


# ----------------------------------------------------------------------
# Feature descriptions
# ----------------------------------------------------------------------

FEATURES = {

    "commit_id": (
        "Commit identifier",
        "Metadata / identifier",
        "Unique identifier of the source commit. Used for joining, "
        "verification, and split identification; never used as a model feature.",
    ),

    "project": (
        "Project",
        "Metadata",
        "Source repository/project associated with the commit.",
    ),

    "buggy": (
        "Buggy label",
        "Target",
        "Binary defect label derived from the canonical SZZ-based labeling "
        "pipeline. 0 = non-buggy, 1 = buggy.",
    ),

    "fix": (
        "Fix indicator",
        "Metadata / source field",
        "Original commit-level fix indicator retained from the canonical dataset.",
    ),

    "year": (
        "Commit year",
        "Metadata",
        "Year associated with the commit author date.",
    ),

    "author_date": (
        "Author date",
        "Temporal metadata",
        "Commit author timestamp used as part of chronological ordering.",
    ),

    "la": (
        "Lines added",
        "JIT metric",
        "Number of lines added by the commit.",
    ),

    "ld": (
        "Lines deleted",
        "JIT metric",
        "Number of lines deleted by the commit.",
    ),

    "nf": (
        "Number of modified files",
        "JIT metric",
        "Number of files modified by the commit.",
    ),

    "nd": (
        "Number of modified directories",
        "JIT metric",
        "Number of directories affected by the commit.",
    ),

    "ns": (
        "Number of modified subsystems",
        "JIT metric",
        "Number of subsystems affected by the commit.",
    ),

    "ent": (
        "Change entropy",
        "JIT metric",
        "Entropy-based measure of how changes are distributed across "
        "the modified files.",
    ),

    "ndev": (
        "Number of developers",
        "JIT metric",
        "Number of developers associated with the relevant project/history "
        "used by the JIT metric computation.",
    ),

    "age": (
        "Change age",
        "JIT metric",
        "Age-related temporal metric associated with the modified code.",
    ),

    "nuc": (
        "Number of unique changes",
        "JIT metric",
        "Historical change-count metric associated with the modified code.",
    ),

    "aexp": (
        "Developer experience",
        "JIT metric",
        "Developer experience measure used by the JIT defect prediction model.",
    ),

    "arexp": (
        "Recent developer experience",
        "JIT metric",
        "Recent developer experience measure used by the JIT model.",
    ),

    "asexp": (
        "Subsystem experience",
        "JIT metric",
        "Developer subsystem-experience measure used by the JIT model.",
    ),

    "message": (
        "Commit message",
        "Source text",
        "Original commit message associated with the change.",
    ),

    "diff": (
        "Code diff",
        "Source text",
        "Original textual representation of the code changes introduced by the commit.",
    ),

    "embedding": (
        "CodeBERT embedding",
        "Semantic representation",
        "768-dimensional CodeBERT representation of the commit/change content. "
        "Stored as one serialized vector column. For the experiments using "
        "PCA, PCA was fitted on training data only and validation/test data "
        "were transformed using the same fitted model.",
    ),

    "intent": (
        "Change intent",
        "LLM categorical semantic feature",
        "LLM-generated semantic classification describing the primary intent "
        "of the change.",
    ),

    "intent_confidence": (
        "Intent confidence",
        "LLM numerical semantic feature",
        "LLM confidence associated with the predicted intent category.",
    ),

    "intent_margin": (
        "Intent margin",
        "LLM numerical semantic feature",
        "Margin associated with the LLM intent prediction.",
    ),

    "change": (
        "Change type",
        "LLM categorical semantic feature",
        "LLM-generated classification describing the type of change.",
    ),

    "change_confidence": (
        "Change confidence",
        "LLM numerical semantic feature",
        "LLM confidence associated with the predicted change category.",
    ),

    "change_margin": (
        "Change margin",
        "LLM numerical semantic feature",
        "Margin associated with the LLM change prediction.",
    ),

    "risk": (
        "Change risk",
        "LLM categorical semantic feature",
        "LLM-generated categorical assessment of change risk.",
    ),

    "risk_confidence": (
        "Risk confidence",
        "LLM numerical semantic feature",
        "LLM confidence associated with the predicted risk category.",
    ),

    "risk_margin": (
        "Risk margin",
        "LLM numerical semantic feature",
        "Margin associated with the LLM risk prediction.",
    ),

    "complexity": (
        "Change complexity",
        "LLM categorical semantic feature",
        "LLM-generated categorical assessment of change complexity.",
    ),

    "complexity_confidence": (
        "Complexity confidence",
        "LLM numerical semantic feature",
        "LLM confidence associated with the predicted complexity category.",
    ),

    "complexity_margin": (
        "Complexity margin",
        "LLM numerical semantic feature",
        "Margin associated with the LLM complexity prediction.",
    ),

    "scope": (
        "Change scope",
        "LLM categorical semantic feature",
        "LLM-generated categorical assessment of the scope of the change.",
    ),

    "scope_confidence": (
        "Scope confidence",
        "LLM numerical semantic feature",
        "LLM confidence associated with the predicted scope category.",
    ),

    "scope_margin": (
        "Scope margin",
        "LLM numerical semantic feature",
        "Margin associated with the LLM scope prediction.",
    ),

    "test": (
        "Testing characteristic",
        "LLM categorical semantic feature",
        "LLM-generated classification describing the testing characteristic "
        "of the change.",
    ),

    "test_confidence": (
        "Test confidence",
        "LLM numerical semantic feature",
        "LLM confidence associated with the predicted testing category.",
    ),

    "test_margin": (
        "Test margin",
        "LLM numerical semantic feature",
        "Margin associated with the LLM testing prediction.",
    ),

    "security": (
        "Security characteristic",
        "LLM categorical semantic feature",
        "LLM-generated classification describing the security characteristic "
        "of the change.",
    ),

    "security_confidence": (
        "Security confidence",
        "LLM numerical semantic feature",
        "LLM confidence associated with the predicted security category.",
    ),

    "security_margin": (
        "Security margin",
        "LLM numerical semantic feature",
        "Margin associated with the LLM security prediction.",
    ),

    "model": (
        "LLM model",
        "Provenance metadata",
        "Identifier of the LLM model used to generate the semantic features.",
    ),

    "schema_version": (
        "LLM feature schema version",
        "Provenance metadata",
        "Version identifier for the LLM semantic-feature schema.",
    ),
}


def main():

    logger.info("=" * 70)
    logger.info("GENERATING FEATURE DICTIONARY")
    logger.info("=" * 70)

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Publication dataset not found:\n{DATASET}"
        )

    logger.info("Loading publication dataset...")

    df = pd.read_csv(
        DATASET,
        nrows=5,
        low_memory=False,
    )

    actual_columns = list(df.columns)

    if actual_columns != EXPECTED_COLUMNS:
        logger.error("Unexpected publication schema.")
        logger.error("Expected %d columns.", len(EXPECTED_COLUMNS))
        logger.error("Found %d columns.", len(actual_columns))

        missing = [
            c for c in EXPECTED_COLUMNS
            if c not in actual_columns
        ]

        extra = [
            c for c in actual_columns
            if c not in EXPECTED_COLUMNS
        ]

        if missing:
            logger.error("Missing columns: %s", missing)

        if extra:
            logger.error("Unexpected columns: %s", extra)

        raise AssertionError(
            "Publication dataset schema does not match expected 44-column schema."
        )

    logger.info(
        "PASS: publication schema contains %d columns.",
        len(actual_columns),
    )

    rows = []

    for index, column in enumerate(actual_columns, start=1):

        if column not in FEATURES:
            raise AssertionError(
                f"No feature definition available for: {column}"
            )

        name, category, description = FEATURES[column]

        rows.append({
            "column_order": index,
            "column": column,
            "feature_name": name,
            "category": category,
            "description": description,
        })

    dictionary = pd.DataFrame(rows)

    dictionary.to_csv(
        CSV_OUTPUT,
        index=False,
        encoding="utf-8",
    )

    logger.info(
        "Saved CSV feature dictionary:\n%s",
        CSV_OUTPUT,
    )

    # ------------------------------------------------------------------
    # Markdown version
    # ------------------------------------------------------------------

    md = []

    md.append("# Publication Dataset Feature Dictionary")
    md.append("")
    md.append(
        "The final publication dataset contains **59,996 commits "
        "and 44 stored columns**."
    )
    md.append("")
    md.append(
        "The 44-column storage schema should not be confused with the "
        "number of model-ready features. The LLM representation contains "
        "21 raw semantic columns (7 categorical + 14 confidence/margin "
        "columns). After one-hot encoding the seven categorical dimensions, "
        "these become 37 categorical features plus 14 numerical features, "
        "for a total of **51 model-ready LLM features**."
    )
    md.append("")

    md.append("## Dataset-Level Feature Summary")
    md.append("")
    md.append("| Representation | Stored columns | Model-ready dimensions |")
    md.append("|---|---:|---:|")
    md.append("| Metadata / identifiers | 2 | 0 |")
    md.append("| Target | 1 | 1 |")
    md.append("| Other source metadata | 3 | 0 |")
    md.append("| JIT metrics | 12 | 12 |")
    md.append("| Commit message | 1 | 0 |")
    md.append("| Code diff | 1 | 0 |")
    md.append("| CodeBERT embedding | 1 serialized vector | 768 / 384* |")
    md.append("| LLM categorical outputs | 7 | 37 after one-hot encoding |")
    md.append("| LLM confidence/margin | 14 | 14 |")
    md.append("| LLM provenance | 2 | 0 |")
    md.append("")
    md.append(
        "*The fusion experiments using CodeBERT apply train-only PCA "
        "from 768 to 384 dimensions."
    )
    md.append("")

    md.append("## Complete Feature Dictionary")
    md.append("")
    md.append(
        "| # | Column | Feature | Category | Description |"
    )
    md.append("|---:|---|---|---|---|")

    for row in rows:
        description = row["description"].replace("|", "\\|")

        md.append(
            f'| {row["column_order"]} '
            f'| `{row["column"]}` '
            f'| {row["feature_name"]} '
            f'| {row["category"]} '
            f'| {description} |'
        )

    md.append("")
    md.append("## LLM Representation")
    md.append("")
    md.append(
        "The stored LLM semantic representation contains seven categorical "
        "dimensions:"
    )
    md.append("")
    md.append(
        "1. Intent  \n"
        "2. Change  \n"
        "3. Risk  \n"
        "4. Complexity  \n"
        "5. Scope  \n"
        "6. Test  \n"
        "7. Security"
    )
    md.append("")
    md.append(
        "Each categorical dimension has an associated confidence and margin, "
        "giving 14 numerical LLM features."
    )
    md.append("")
    md.append(
        "**Raw LLM semantic columns:** 7 categorical + 14 numerical = "
        "21 columns."
    )
    md.append("")
    md.append(
        "**Model-ready LLM representation:** 37 one-hot categorical "
        "features + 14 numerical confidence/margin features = "
        "**51 features**."
    )
    md.append("")
    md.append(
        "The `model` and `schema_version` columns are provenance metadata "
        "and are not used as predictive features."
    )
    md.append("")

    md.append("## Model Feature Counts")
    md.append("")
    md.append("| Experiment representation | Feature dimensions |")
    md.append("|---|---:|")
    md.append("| JIT only | 12 |")
    md.append("| CodeBERT only | 384 |")
    md.append("| LLM only | 51 |")
    md.append("| JIT + CodeBERT | 396 |")
    md.append("| JIT + LLM | 63 |")
    md.append("| CodeBERT + LLM | 435 |")
    md.append("| JIT + CodeBERT + LLM | 447 |")
    md.append("| Exp. 8 compact LLM | 410 |")
    md.append("| Exp. 9 selected LLM | 410 |")
    md.append("")

    MD_OUTPUT.write_text(
        "\n".join(md),
        encoding="utf-8",
    )

    logger.info(
        "Saved Markdown feature dictionary:\n%s",
        MD_OUTPUT,
    )

    logger.info("=" * 70)
    logger.info("FEATURE DICTIONARY GENERATION PASSED")
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        logger.error("FEATURE DICTIONARY GENERATION FAILED")
        logger.error("%s", error)
        sys.exit(1)