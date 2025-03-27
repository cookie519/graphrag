import pandas as pd

# Read the CSV file
file_path = '/projects/JHA/shared/triples/32k_lim4_rel32.csv'
df = pd.read_csv(file_path)
print(df.head())
print(df.shape)

