"""
Multilanguage Dataset Inspection Script

Purpose:
    Perform a sanity check on the final multilanguage dataset by
    displaying dataset dimensions, column names, and sample records
    from different sections of the file.

Checks performed:
    - Dataset shape
    - Available columns/features
    - First 5 rows
    - Sample rows from the middle of the dataset
    - Last 5 rows

Input:
    - final_multilanguage.csv
      or
    - final_multilanguage_dataset.csv

Output:
    Console-based dataset summary for verification and debugging.

Notes:
    Intended for dataset validation before feature extraction,
    model training, or experimental analysis.
"""
import pandas as pd
from pathlib import Path

pd.set_option('display.max_columns', None)

print("Loading dataset...")
dataset_path = Path("final_multilanguage.csv")
if not dataset_path.exists():
    dataset_path = Path("final_multilanguage_dataset.csv")

print(f"Reading from: {dataset_path}")
df = pd.read_csv(dataset_path)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

# ✅ 1. First 5 rows (all features)
print("\nFirst 5 rows of the dataset:")
print(df.head(5))

# ✅ 2. 5 rows skipping last 1000 (for Python entries)
if len(df) > 1000:
    print("\n5 rows after skipping last 1000 rows (for inspection):")
    print(df.iloc[:-1100].tail(5))
else:
    print("\nDataset too small to skip 1000 rows.")

# ✅ 3. Last 5 rows
print("\nLast 5 rows of the dataset:")
print(df.tail(5))