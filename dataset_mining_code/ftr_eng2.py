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
