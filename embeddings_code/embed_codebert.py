"""
CodeBERT Embedding Extraction Pipeline

Purpose:
    Generate semantic embeddings for commit messages and code diffs
    using Microsoft's CodeBERT model for software defect prediction.

Workflow:
    1. Load the multilingual defect dataset.
    2. Combine commit messages and code diffs into a single input text.
    3. Tokenize inputs using CodeBERT tokenizer.
    4. Generate CLS embeddings using the pretrained CodeBERT model.
    5. Save embeddings alongside original dataset features.
    6. Maintain progress checkpoints for safe resumption after interruptions.

Input:
    - final_multilanguage_dataset.csv

Output:
    - final_multilanguage_dataset_with_embeddings.csv

Features:
    - GPU acceleration (CUDA support)
    - Automatic batch processing
    - Resume from checkpoints after crashes/interruption
    - Out-of-memory recovery by reducing batch size
    - Disk space validation before execution
    - Progress tracking for large datasets

Use Case:
    Converts textual commit information into dense vector
    representations that can be used as input features for
    machine learning and deep learning based defect prediction models.
"""
import os
import argparse
import json
import csv
import shutil
from pathlib import Path
from typing import List, Tuple, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm


class TextDataset(Dataset):
    def __init__(self, items: List[Tuple[int, str]]):
        # items: list of (orig_index, text)
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def identity_collate(batch):
    return batch


def load_data(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    # Ensure consistent row order and index
    df.reset_index(drop=True, inplace=True)
    # Fill missing
    df['diff'] = df.get('diff', '').fillna('')
    df['message'] = df.get('message', '').fillna('')
    # Combine
    df['input_text'] = df['message'].astype(str) + ' [SEP] ' + df['diff'].astype(str)
    return df


def preprocess_batch(batch_items, tokenizer, device):
    # batch_items: list of (orig_index, text)
    idxs, texts = zip(*batch_items)
    enc = tokenizer(list(texts), padding=True, truncation=True, max_length=256, return_tensors='pt')
    return idxs, enc


def _load_progress(progress_path: str, output_path: str = None) -> Dict[str, int]:
    # If output file is absent, ignore stale progress to avoid skipping unwritten rows.
    if output_path is not None and not os.path.exists(output_path):
        return {}
    if os.path.exists(progress_path):
        try:
            with open(progress_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {int(k): v for k, v in data.items()}
        except Exception:
            return {}
    return {}


def _save_progress(progress_path: str, processed_map: Dict[int, int]):
    os.makedirs(os.path.dirname(os.path.abspath(progress_path)) or '.', exist_ok=True)
    tmp = progress_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in processed_map.items()}, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, progress_path)


def _init_output(output_path: str, df_columns: List[str]):
    # Ensure parent exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    if os.path.exists(output_path):
        return
    # write header
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(list(df_columns) + ['embedding'])


def get_embeddings(model, tokenizer, items: List[Tuple[int, str]], df: pd.DataFrame, output_path: str,
                   device, batch_size: int, num_workers: int, pin_memory: bool,
                   progress_path: str, checkpoint_interval: int = 1000):
    """
    Stream embeddings to `output_path` as they are produced and maintain a small progress JSON at `progress_path`.
    Returns the processed indices map (index -> 1).
    This avoids accumulating large checkpoint files and enables safe resume.
    """

    processed_map = _load_progress(progress_path, output_path=output_path)

    dataset = TextDataset(items)
    current_batch_size = batch_size

    # On Windows multiprocessing requires picklable objects; fall back to single-process if needed
    if os.name == 'nt' and num_workers > 0:
        print('Windows detected: forcing num_workers=0 to avoid multiprocessing pickling issues')
        num_workers = 0

    def make_loader(bs):
        return DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=num_workers,
                          pin_memory=pin_memory, collate_fn=identity_collate)

    loader = make_loader(current_batch_size)
    total_items = len(items)
    pbar = tqdm(total=total_items, desc='Embedding', unit='rows')

    _init_output(output_path, df.columns)

    try:
        for batch in loader:
            idxs, enc = preprocess_batch(batch, tokenizer, device)
            for k, v in enc.items():
                enc[k] = v.to(device)

            with torch.no_grad():
                outputs = model(**enc)
                cls_emb = outputs.last_hidden_state[:, 0, :]
                if device.type == 'cuda':
                    cls_emb = cls_emb.half().cpu()
                else:
                    cls_emb = cls_emb.cpu()

            emb_list = cls_emb.float().tolist()

            rows_to_write = []
            indices_to_mark = []
            for idx, emb in zip(idxs, emb_list):
                if int(idx) in processed_map:
                    continue
                # get original row values as strings
                row = df.iloc[int(idx)].tolist()
                # store embedding as JSON string to keep CSV compact and parseable
                rows_to_write.append(row + [json.dumps(emb)])
                indices_to_mark.append(int(idx))

            # Append batch to output file
            if rows_to_write:
                try:
                    with open(output_path, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerows(rows_to_write)
                        f.flush()
                        os.fsync(f.fileno())
                    # Mark as processed only after successful write.
                    for idx in indices_to_mark:
                        processed_map[idx] = 1
                except OSError as e:
                    if getattr(e, 'errno', None) == 28 or 'No space' in str(e):
                        print('Disk full while writing output. Saving progress and exiting.')
                        _save_progress(progress_path, processed_map)
                        raise
                    else:
                        raise

            # periodically save progress
            if len(processed_map) % checkpoint_interval == 0:
                _save_progress(progress_path, processed_map)

            pbar.update(len(idxs))

    except RuntimeError as e:
        msg = str(e).lower()
        if 'out of memory' in msg or 'cuda' in msg:
            print(f"CUDA OOM encountered with batch_size={current_batch_size}. Reducing batch size and retrying.")
            torch.cuda.empty_cache()
            if current_batch_size == 1:
                _save_progress(progress_path, processed_map)
                raise
            current_batch_size = max(1, current_batch_size // 2)
            loader = make_loader(current_batch_size)
            # continue processing with smaller batches
        else:
            _save_progress(progress_path, processed_map)
            raise

    finally:
        _save_progress(progress_path, processed_map)
        pbar.close()

    return processed_map


def save_part(checkpoint_dir: str, start_idx: int, end_idx: int, idxs: List[int], embeddings: List[List[float]]):
    part_df = pd.DataFrame({'index': list(idxs), 'embedding': embeddings})
    fname = os.path.join(checkpoint_dir, f'part_{start_idx}_{end_idx}.csv')
    part_df.to_csv(fname, index=False)


def load_checkpoint(checkpoint_dir: str):
    # returns mapping index->embedding from all parts
    if not os.path.isdir(checkpoint_dir):
        return {}
    parts = sorted(Path(checkpoint_dir).glob('part_*.csv')) + sorted(Path(checkpoint_dir).glob('part_*.parquet'))
    if not parts:
        return {}
    mapping = {}
    for p in parts:
        try:
            if p.suffix.lower() == '.parquet':
                dfp = pd.read_parquet(p)
            else:
                dfp = pd.read_csv(p)
            for idx, emb in zip(dfp['index'].tolist(), dfp['embedding'].tolist()):
                mapping[int(idx)] = emb
        except Exception as e:
            print(f"Warning: failed to read checkpoint part {p}: {e}")
    return mapping


def assemble_and_save_final(df: pd.DataFrame, processed_map: Dict[int, int], output_path: str):
    # processed_map: mapping of written indices
    written = len(processed_map)
    total = len(df)
    print(f"Assembled output: {written} / {total} rows written to {output_path}")
    if written < total:
        print(f"Warning: {total - written} rows missing embeddings. Resume later to finish them.")


def main():
    parser = argparse.ArgumentParser(description='Resumable CodeBERT embedding extraction')
    parser.add_argument('--input', type=str, default='./final_multilanguage_dataset.csv')
    parser.add_argument('--output', type=str, default='./final_multilanguage_dataset_with_embeddings.csv')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_codebert')
    parser.add_argument('--checkpoint_interval', type=int, default=1000)
    parser.add_argument('--min_free_gb', type=float, default=5.0,
                        help='Minimum free GB required on output drive before starting')
    parser.add_argument('--max_rows', type=int, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    df = load_data(args.input)
    total_rows = len(df)
    print(f"Loaded {total_rows} rows from {args.input}")

    # Disk space pre-check for output path
    out_dir = os.path.dirname(os.path.abspath(args.output)) or '.'
    try:
        du = shutil.disk_usage(out_dir)
        free_gb = du.free / (1024 ** 3)
    except Exception:
        free_gb = 0.0
    if free_gb < args.min_free_gb:
        print(f"ERROR: Not enough free space on {out_dir} ({free_gb:.2f} GB). Require at least {args.min_free_gb} GB. Aborting.")
        return

    tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
    model = AutoModel.from_pretrained('microsoft/codebert-base')
    model.eval()

    if device.type == 'cuda':
        # Move model to GPU and attempt FP16 for speed/memory
        try:
            model.to(device)
            model.half()
        except Exception:
            # fallback: ensure model is on device (keep fp32 if half fails)
            model.to(device)
        # Print GPU info for verification
        try:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    # Build list of items to process: (orig_index, text)
    items = []
    for i, txt in enumerate(df['input_text'].tolist()):
        if args.max_rows is not None and i >= args.max_rows:
            break
        items.append((i, txt))

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    progress_path = os.path.join(args.checkpoint_dir, f'progress_{Path(args.output).stem}.json')
    if items:
        print(f"Processing {len(items)} rows (batch_size={args.batch_size})")
        processed_map = get_embeddings(model, tokenizer, items, df, args.output, device,
                                       args.batch_size, args.num_workers, True, progress_path,
                                       args.checkpoint_interval)
    else:
        print('No rows to process.')
        processed_map = {}

    # Assemble final file (report)
    assemble_and_save_final(df, processed_map, args.output)


if __name__ == '__main__':
    main()
