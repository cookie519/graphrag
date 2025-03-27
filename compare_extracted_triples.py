import re
import pandas as pd
from io import StringIO

def extract_entities(text):
    pattern = r"-----Entities-----(.*?)-----Relationships-----"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None

def extract_relations(text):
    pattern = r"-----Relationships-----(.*?)---Goal---"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None

# Define function to read data
def read_entities(text):
    df = pd.read_csv(StringIO(text), sep="|", dtype=str)
    df["entity"] = df["entity"].str.strip()
    return set(df["entity"])

def count_relations(text):
    df = pd.read_csv(StringIO(text), sep="|", dtype=str)
    return len(df)

injected_file = "/scratch/gpfs/jx0800/data/graphrag/results/injected_32k_query_output_8B.log"
expanded_file = "/scratch/gpfs/jx0800/data/graphrag/results/expanded_32k_query_output_8B.log"

with open(injected_file, 'r') as file:
    injected = file.read()

injected = injected.strip().split('-'*50)
injected = injected[:-1] 

with open(expanded_file, 'r') as file:
    expanded = file.read()

expanded = expanded.strip().split('-'*50)
expanded = expanded[:-1]

nums = {'inj': 0, 'exp': 0, 'overlap': 0, 'relations_inj': 0, 'relations_exp': 0}

# Process each response
for inj, exp in zip(injected, expanded):
    inj_entities = extract_entities(inj)
    exp_entities = extract_entities(exp)

    # Read entities from A and B
    entities_inj = read_entities(inj_entities)
    entities_exp = read_entities(exp_entities)

    # Count relations
    num_relations_inj = count_relations(extract_relations(inj))
    num_relations_exp = count_relations(extract_relations(exp))
    nums['relations_inj'] += num_relations_inj
    nums['relations_exp'] += num_relations_exp

    # Compute overlap
    common_entities = entities_inj.intersection(entities_exp)
    nums['inj'] += len(entities_inj)
    nums['exp'] += len(entities_exp)
    nums['overlap'] += len(common_entities)

print(f"Total entities in injected: {nums['inj']}")
print(f"Total entities in expanded: {nums['exp']}")
print(f"Total overlapping entities: {nums['overlap']}")
print(f"Overlap percentage: {nums['overlap'] / nums['inj']:.2%}")
print(f"Overlap percentage: {nums['overlap'] / nums['exp']:.2%}")

print(f"Total relations in injected: {nums['relations_inj']}")
print(f"Total relations in expanded: {nums['relations_exp']}")