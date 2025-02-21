import pyarrow.parquet as pq
import os
import pandas as pd

# # Load the Parquet file
# table = pq.read_table("/scratch/gpfs/jx0800/data/graphrag/output/text_units.parquet")  # Replace with your actual file path

# # Print the table schema
# print("column of the Parquet file:")
# print(table.column_names)


# Directory containing Parquet files
parquet_dir = "/scratch/gpfs/jx0800/data/graphrag/output_api"

# Get list of all .parquet files in the directory
parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith(".parquet")]

# Read and display each Parquet file
for file in parquet_files:
    file_path = os.path.join(parquet_dir, file)
    table = pq.read_table(file_path)  # Read parquet file

    print(f"Displaying contents of: {file}")
    print(table.column_names)
    print(table.to_pandas().head())  # Display first 5 rows of the table