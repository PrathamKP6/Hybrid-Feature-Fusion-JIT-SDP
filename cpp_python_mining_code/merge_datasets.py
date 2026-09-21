import pandas as pd
from pathlib import Path

base = Path(__file__).resolve().parent.parent / 'data'
java_df = pd.read_csv(base / 'java_dataset.csv')
python_df = pd.read_csv(Path(__file__).resolve().parent / 'python_flask_dataset_szz.csv')
opencv_df = pd.read_csv(Path(__file__).resolve().parent / 'cpp_opencv_dataset_szz.csv')

final_df = pd.concat([java_df, python_df, opencv_df], ignore_index=True)
final_df.to_csv(base / 'final_multilanguage_dataset.csv', index=False)

print(f'✓ Final multilanguage dataset: {final_df.shape[0]} rows × {final_df.shape[1]} columns')
print(f'Projects: {sorted(final_df["project"].unique().tolist())}')
print(f'Buggy: {int((final_df["buggy"] == 1).sum())} True, {int((final_df["buggy"] == 0).sum())} False')
