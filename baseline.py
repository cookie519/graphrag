import torch
from vllm import LLM, SamplingParams
from datasets import Dataset
import json
import ast
import os
import logging
import pandas as pd

task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
task_id = 0 if task_id is None else int(task_id)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# model_id = "/projects/JHA/jx0800/models/Llama-3.3-70B-Instruct" 
# model_name = 'llama'
model_id = "/scratch/gpfs/JHA/models/Llama-3.3-70B-Instruct" #"/scratch/gpfs/jx0800/models/Llama-3.1-8B-Instruct" #"/projects/JHA/jx0800/models/DeepSeek-R1-Distill-Llama-70B" 

# File paths
query_file = "/projects/JHA/shared/dataset/mcq_filtered_by_match/queries.txt"

logger.info('Loading LLM')
llm = LLM(model=model_id, trust_remote_code=False, dtype=torch.bfloat16, tensor_parallel_size=4, quantization="fp8") # , quantization="fp8" tensor_parallel_size=4, 
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=100, max_tokens=1024)


system_prompt = r"""---Role---

You are a helpful assistant responding to questions about data in the tables provided.

---Target response format---

Answer in the following format:
Analysis of the question: 
- Understand the question and find out what is needed to answer it
- Analyze and answer the question

Answer:
- Provide a concise answer using `\boxed{}`, select only the correct letter (e.g., \boxed{C}) 
"""

batch_size = 200
num = 5
start_idx = task_id * batch_size * num
stop_idx = start_idx + batch_size * num

# We'll collect tail predictions in a dictionary mapping head -> list of tails.
messages = []
responses = []

# Read queries from the file
with open(query_file, "r") as file:
    queries = file.readlines()

# Process each query and save responses
for i, query in enumerate(queries):
    if i < start_idx:
        continue
    if i == stop_idx:
        break

    query = query.strip()  # Remove leading/trailing whitespace
    logger.info(f"Processing query {i}:")
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "text", "text": query}]}
    ]

    # Build the current batch
    if i % batch_size == 0:
        messages = [message]
    else:
        messages.append(message)

    # When the batch is complete or we're at the end of the dataset, process it.
    if i % batch_size == batch_size - 1 or i == len(queries) - 1:
        outputs = llm.chat(messages, sampling_params=sampling_params, use_tqdm=False)
        for out in outputs:
            response = out.outputs[0].text
            responses.append(response)             


response_file = "/scratch/gpfs/jx0800/data/graphrag/results/baseline_responses_llama8B.log"
with open(response_file, "w") as output_file:
    for i, query in enumerate(queries):
        output_file.write(f"Query {i}: {query}\n")
        output_file.write(f"Response: {responses[i]}\n")
        output_file.write("-" * 50 + "\n")


