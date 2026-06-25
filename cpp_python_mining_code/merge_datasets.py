import pandas as pd
from pathlib import Path

base = Path('.')
java_df = pd.read_csv('java_dataset.csv')
python_df = pd.read_csv('python_flask_dataset_szz.csv')
opencv_df = pd.read_csv('cpp_opencv_dataset_szz.csv')

final_df = pd.concat([java_df, python_df, opencv_df], ignore_index=True)
output_path = Path('data') / 'final_multilanguage_dataset.csv'
output_path.parent.mkdir(parents=True, exist_ok=True)
final_df.to_csv(output_path, index=False)

print(f'✓ Final multilanguage dataset: {final_df.shape[0]} rows × {final_df.shape[1]} columns')
print(f'Saved: {output_path}')
print(f'Projects: {sorted(final_df["project"].unique().tolist())}')
print(f'Buggy: {int((final_df["buggy"] == 1).sum())} True, {int((final_df["buggy"] == 0).sum())} False')
