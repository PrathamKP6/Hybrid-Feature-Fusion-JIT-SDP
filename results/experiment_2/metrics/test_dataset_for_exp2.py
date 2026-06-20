import pandas as pd
import numpy as np
import ast

# Load frozen sample
df = pd.read_csv('results/frozen_sample_dataset.csv', nrows=100)

# Parse embedding column
def parse_embedding(x):
    if pd.isna(x):
        return None
    if isinstance(x, str):
        return ast.literal_eval(x)
    return x

embeddings = df['embedding'].apply(parse_embedding)
emb_matrix = np.vstack(embeddings.dropna().values)

print("Shape:", emb_matrix.shape)
print("Data type:", emb_matrix.dtype)
print("Min/Max:", emb_matrix.min(), emb_matrix.max())
print("Mean:", emb_matrix.mean(axis=0)[:10])  # First 10 dims
print("Std Dev:", emb_matrix.std(axis=0)[:10])
print("Sparsity (< 0.01):", np.sum(np.abs(emb_matrix) < 0.01) / emb_matrix.size)