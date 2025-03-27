from datasets import Dataset

# load triples
triples_file = "/projects/JHA/shared/triples/50k_seq_lim2_unique_triples"
dataset = Dataset.load_from_disk(triples_file)

# form prompt; each example has head (str), relation (str), and tails (a list of str).
prompts = []
for example in dataset:
    for tail in example['tails']:
        prompt = f"Evaluate this triple: {example['head']}, {example['relation']}, {tail}. Is this triple valid or not? Respond with reason, finally answer \\boxed{{True}} or \\boxed{{False}}"
        prompts.append(prompt)

# save prompts to a file
query_file = "/home/jx0800/graphrag/50k_seq_lim2_unique_triples_prompts.txt"
with open(query_file, "w") as file:
    for prompt in prompts:
        file.write(prompt + "\n")

# # load prompts
# with open(query_file, "r") as file:
#     prompts = file.readlines()

