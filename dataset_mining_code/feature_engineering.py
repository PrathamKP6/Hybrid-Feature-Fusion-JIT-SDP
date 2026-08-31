"""
Dataset Inspection Script

Purpose:
    Perform a quick sanity check on final_dataset.csv by displaying
    dataset dimensions, column names, sample records, available projects,
    and the last few rows.

Checks performed:
    - Dataset shape (rows × columns)
    - Feature/column names
    - First few records
    - Unique project names
    - Last 10 records

Output:
    Printed summary information in the console.
"""
import pandas as pd
pd.set_option('display.max_columns', None)

print("Loading dataset...")
df = pd.read_csv("final_dataset.csv")

print(df.shape)
print(df.columns)
print(df.head())
print(df["project"].unique())

# Show the last 10 rows of the dataset
print("\nLast 10 rows of the dataset:")
print(df.tail(10))
