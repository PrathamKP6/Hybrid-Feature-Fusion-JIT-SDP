import pandas as pd
import ast
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent.parent / "data"

datasets = {
    "CHRONOLOGICAL TRAIN":
        BASE / "chronological_split_dataset" / "train.csv",

    "CHRONOLOGICAL VALIDATION":
        BASE / "chronological_split_dataset" / "validation.csv",

    "CHRONOLOGICAL TEST":
        BASE / "chronological_split_dataset" / "test.csv",

    "PCA TRAIN":
        BASE / "dataset_with_pca" / "train.csv",

    "PCA VALIDATION":
        BASE / "dataset_with_pca" / "validation.csv",

    "PCA TEST":
        BASE / "dataset_with_pca" / "test.csv",
}


OUTPUT_FILE = Path(__file__).resolve().parent / "dataset_inspection_output.txt"


# ============================================================
# EMBEDDING PARSER
# ============================================================

def parse_embedding(value):

    if isinstance(value, (list, tuple)):
        return list(value)

    try:
        return ast.literal_eval(str(value))
    except Exception:
        return None


# ============================================================
# INSPECTION
# ============================================================

def inspect_dataset(name, path, terminal_output, file_output):

    if not path.exists():

        terminal_output.append(
            f"{name}: FILE NOT FOUND\n{path}\n"
        )

        file_output.append(
            f"\n{name}\n{'=' * 70}\nFILE NOT FOUND: {path}\n"
        )

        return

    df = pd.read_csv(path, low_memory=False)

    rows, columns = df.shape

    # --------------------------------------------------------
    # FIND EMBEDDING COLUMN
    # --------------------------------------------------------

    embedding_columns = [
        c for c in df.columns
        if "embedding" in c.lower()
    ]

    embedding_column = (
        embedding_columns[0]
        if embedding_columns
        else None
    )

    embedding_dimension = "N/A"

    if embedding_column and len(df) > 0:

        embedding = parse_embedding(
            df[embedding_column].iloc[0]
        )

        if embedding is not None:
            embedding_dimension = len(embedding)

    # ========================================================
    # TERMINAL OUTPUT
    # ========================================================

    terminal_output.append(
        f"\n{'=' * 70}\n"
        f"{name}\n"
        f"{'=' * 70}\n"
        f"Shape: ({rows}, {columns})\n\n"
        f"Columns ({columns}):\n"
    )

    for i, column in enumerate(df.columns, 1):
        terminal_output.append(
            f"{i}. {column}\n"
        )

    terminal_output.append(
        f"\nEmbedding column: {embedding_column}\n"
        f"Embedding dimension: {embedding_dimension}\n"
    )

    # Show only first 10 embedding values in terminal
    if embedding_column and len(df) > 0:

        terminal_output.append(
            "\nFirst 2 rows — summary:\n"
        )

        for row_number in range(min(2, len(df))):

            row = df.iloc[row_number]

            terminal_output.append(
                f"\nROW {row_number + 1}\n"
                f"{'-' * 50}\n"
            )

            for column in df.columns:

                if column == embedding_column:
                    embedding = parse_embedding(row[column])

                    if embedding is not None:
                        terminal_output.append(
                            f"embedding shape: ({len(embedding)},)\n"
                            f"embedding first 10 values: "
                            f"{embedding[:10]}\n"
                        )
                    else:
                        terminal_output.append(
                            "embedding: Could not parse\n"
                        )

                elif column == "diff":

                    diff = str(row[column])

                    terminal_output.append(
                        f"diff: {diff[:300]}"
                        + ("..." if len(diff) > 300 else "")
                        + "\n"
                    )

                else:

                    terminal_output.append(
                        f"{column}: {row[column]}\n"
                    )

    # ========================================================
    # COMPLETE OUTPUT TO FILE
    # ========================================================

    file_output.append(
        f"\n{'=' * 80}\n"
        f"{name}\n"
        f"{'=' * 80}\n"
        f"FILE: {path}\n"
        f"Shape: ({rows}, {columns})\n\n"
    )

    file_output.append(
        f"Columns ({columns}):\n"
    )

    for i, column in enumerate(df.columns, 1):
        file_output.append(
            f"{i}. {column}\n"
        )

    file_output.append(
        f"\nEmbedding column: {embedding_column}\n"
        f"Embedding dimension: {embedding_dimension}\n"
    )

    file_output.append(
        "\n\nCOMPLETE CONTENT OF FIRST 2 ROWS\n"
    )

    # --------------------------------------------------------
    # COMPLETE FIRST 2 ROWS
    # --------------------------------------------------------

    for row_number in range(min(2, len(df))):

        row = df.iloc[row_number]

        file_output.append(
            f"\n{'-' * 80}\n"
            f"ROW {row_number + 1}\n"
            f"{'-' * 80}\n"
        )

        for column in df.columns:

            if column == embedding_column:

                embedding = parse_embedding(row[column])

                file_output.append(
                    f"\n{column}:\n"
                    f"Shape: ({len(embedding)},)\n"
                    f"Values:\n"
                    f"{embedding}\n"
                )

            else:

                file_output.append(
                    f"\n{column}:\n"
                    f"{row[column]}\n"
                )


# ============================================================
# MAIN
# ============================================================

terminal_output = []
file_output = []

terminal_output.append(
    "\n"
    "DATASET INSPECTION\n"
    "=" * 70
    + "\n"
)

file_output.append(
    "DATASET INSPECTION — COMPLETE OUTPUT\n"
    + "=" * 80
    + "\n"
)

for name, path in datasets.items():

    inspect_dataset(
        name,
        path,
        terminal_output,
        file_output
    )


# ============================================================
# WRITE COMPLETE OUTPUT
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.writelines(file_output)


# ============================================================
# TERMINAL
# ============================================================

terminal_output.append(
    "\n"
    + "=" * 70
    + "\n"
    "INSPECTION COMPLETE\n"
    + "=" * 70
    + "\n"
    f"\nComplete first-2-row output saved to:\n"
    f"{OUTPUT_FILE}\n"
)

print("".join(terminal_output))