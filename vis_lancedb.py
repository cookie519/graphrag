import lance

# Path to the Lance dataset
dataset_path = "/scratch/gpfs/jx0800/data/graphrag/output/lancedb/default-entity-description.lance"

# Open the dataset
dataset = lance.dataset(dataset_path)

# Display the schema of the dataset
print(dataset.schema)

# Iterate over the dataset and print the first few rows
for batch in dataset.to_batches():
    print(batch.to_pandas())
    break  # Just print the first batch for demonstration