import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def parse_embedding(value) -> np.ndarray:
    if isinstance(value, str):
        return np.asarray(json.loads(value), dtype=np.float32)
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=np.float32)
    raise ValueError(f"Unsupported embedding cell type: {type(value)}")


def load_embedding_sample(
    csv_path: str,
    sample_size: int,
    random_state: int,
    usecols: Tuple[str, ...] = ("embedding",),
) -> np.ndarray:
    df = pd.read_csv(csv_path, usecols=list(usecols), nrows=sample_size)
    if df.empty:
        raise ValueError(f"No embeddings loaded from {csv_path}.")

    embeddings = [parse_embedding(value) for value in df["embedding"]]
    return np.vstack(embeddings)


def fit_pca(X: np.ndarray, n_components: int) -> Dict[str, np.ndarray]:
    mean_vector = X.mean(axis=0)
    centered = X - mean_vector
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    return {"mean": mean_vector, "components": components}


def save_pca_model(model_path: str, model: Dict[str, np.ndarray]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(model_path)) or ".", exist_ok=True)
    np.savez_compressed(model_path, mean=model["mean"], components=model["components"])


def load_pca_model(model_path: str) -> Dict[str, np.ndarray]:
    data = np.load(model_path)
    return {"mean": data["mean"], "components": data["components"]}


def transform_embeddings(embeddings: np.ndarray, model: Dict[str, np.ndarray]) -> np.ndarray:
    return (embeddings - model["mean"]) @ model["components"].T


def serialize_embedding(embedding: np.ndarray) -> str:
    return json.dumps(embedding.astype(float).tolist())


def atomic_write_json(path: str, obj: Dict) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def load_checkpoint(checkpoint_path: str) -> Dict:
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_completed_chunk": -1, "processed_rows": 0}


def write_header_if_missing(output_path: str, columns) -> None:
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(columns)


def append_chunk_to_csv(output_path: str, df_chunk: pd.DataFrame) -> None:
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        df_chunk.to_csv(f, header=False, index=False)


def process_full_dataset(
    input_path: str,
    output_path: str,
    model: Dict[str, np.ndarray],
    checkpoint_path: str,
    chunk_size: int,
    overwrite: bool,
) -> None:
    checkpoint = load_checkpoint(checkpoint_path)
    last_completed_chunk = checkpoint.get("last_completed_chunk", -1)
    rows_written = checkpoint.get("processed_rows", 0)

    if overwrite and os.path.exists(output_path):
        os.remove(output_path)
        checkpoint = {"last_completed_chunk": -1, "processed_rows": 0}
        last_completed_chunk = -1
        rows_written = 0
        atomic_write_json(checkpoint_path, checkpoint)

    reader = pd.read_csv(input_path, chunksize=chunk_size)

    for chunk_index, chunk in enumerate(reader):
        if chunk_index <= last_completed_chunk:
            continue

        if "embedding" not in chunk.columns:
            raise ValueError("Input CSV must contain an 'embedding' column.")

        embeddings = np.vstack([parse_embedding(value) for value in chunk["embedding"]])
        reduced = transform_embeddings(embeddings, model)

        chunk["embedding"] = [serialize_embedding(vec) for vec in reduced]

        if chunk_index == 0 and not os.path.exists(output_path):
            write_header_if_missing(output_path, chunk.columns)
        elif chunk_index == 0 and os.path.exists(output_path) and rows_written == 0:
            write_header_if_missing(output_path, chunk.columns)

        append_chunk_to_csv(output_path, chunk)

        rows_written += len(chunk)
        checkpoint = {
            "last_completed_chunk": chunk_index,
            "processed_rows": rows_written,
            "last_chunk_size": len(chunk),
        }
        atomic_write_json(checkpoint_path, checkpoint)
        print(f"Completed chunk {chunk_index}: wrote {len(chunk)} rows (total {rows_written}).")

    print(f"Finished PCA transform. Output written to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply PCA reduction to embedding vectors in a CSV file.")
    parser.add_argument("--input", type=str, default="final_multilanguage_dataset_with_embeddings.csv", help="Path to the source CSV file.")
    parser.add_argument("--output", type=str, default="final_multilanguage_dataset_with_embeddings_pca384.csv", help="Path for the output CSV with reduced embeddings.")
    parser.add_argument("--components", type=int, default=384, help="Number of PCA components to keep.")
    parser.add_argument("--sample-size", type=int, default=10000, help="Number of embeddings sampled to fit PCA.")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Number of rows to process per chunk when transforming the full dataset.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pca_transform_progress.json", help="Path to the checkpoint JSON file.")
    parser.add_argument("--model-path", type=str, default="checkpoints/pca_384_model.npz", help="Path to save or load the PCA model.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the existing output and restart from scratch.")
    args = parser.parse_args()

    print("Fitting PCA model from sampled embeddings...")
    if os.path.exists(args.model_path) and not args.overwrite:
        model = load_pca_model(args.model_path)
        print(f"Loaded existing PCA model from {args.model_path}.")
    else:
        sample = load_embedding_sample(args.input, args.sample_size, random_state=42)
        model = fit_pca(sample, args.components)
        save_pca_model(args.model_path, model)
        print(f"Saved PCA model to {args.model_path}.")

    process_full_dataset(
        input_path=args.input,
        output_path=args.output,
        model=model,
        checkpoint_path=args.checkpoint,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
    )

    print("PCA reduction complete.")
    print("If you want a different sample size or component count, re-run with --sample-size or --components.")


if __name__ == "__main__":
    main()
