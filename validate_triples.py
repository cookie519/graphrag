import torch
from vllm import LLM, SamplingParams
from datasets import Dataset
import json
import ast
import os
import logging
import numpy as np

task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
task_id = 0 if task_id is None else int(task_id)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# load prompts
prompt_file = "/home/jx0800/graphrag/50k_seq_lim2_unique_triples_prompts.txt"
with open(prompt_file, "r") as file:
    prompts = file.readlines()

# select prompts idx based on task_id
batch_size = 1000
num = 400
start_idx = task_id * batch_size * num
stop_idx = start_idx + batch_size * num 
stop_idx = len(prompts) if stop_idx > len(prompts) else stop_idx
prompts = prompts[start_idx:stop_idx]
logger.info(f"batch size: {batch_size}")
logger.info(f"num: {num}")
logger.info(f"start_idx: {start_idx}")
logger.info(f"stop_idx: {stop_idx}")

# model
model_id = "/scratch/gpfs/JHA/models/Llama-3.1-8B-Instruct" 

logger.info('Loading LLM')
llm = LLM(model=model_id, trust_remote_code=False, tensor_parallel_size=2, dtype=torch.bfloat16) # , quantization="fp8" tensor_parallel_size=4, 
sampling_params = SamplingParams(temperature=0.0, top_p=0.95, top_k=100, max_tokens=1024)

system_prompt = r"""Role  
You are a medical intelligence expert specializing in diabetes and its comorbidities. Your expertise includes advanced knowledge of medical terminology, diabetes pathophysiology, and its complications. You rigorously evaluate the validity of relationships in medical knowledge graphs.  

Task  
Evaluate whether a given triple (structured as `head, brings about, tail`) is valid or invalid based on these criteria:  
1. Incorrect: Factually wrong relationship (e.g., `diabetes, brings about, dental neuropathic`).  
2. Vague: Terms lack specificity (e.g., `diabetes, brings about, blood disorder`).  
3. Incomplete: Terms are nonspecific or truncated (e.g., `diabetes, brings about, dental`).  
4. Irrelevant: Tail term has no established connection to diabetes or its complications (e.g., `diabetes, brings about, bone`).  

Validation Process  
- Verify the triple using medical knowledge about diabetes and its comorbidities.  
- Check if the triple is: 1) Incorrect, 2) Vague, 3) Incomplete, or 4) Irrelevant. If none of these apply, the triple is valid.
- Provide a clear rationale for your decision.
- Conclude with \boxed{True} for valid triples or \boxed{False} for invalid triples.

Examples 
Valid:  
- `type 2 diabetes mellitus, brings about, atherosclerosis`  
- `insulin, brings about, hypophosphatemia`
- `insulin,brings about,blood glucose decreased`
- `obesity,brings about,coronary artery thrombosis`

Invalid:
- `diabetes, brings about, dental neuropathic` (incorrect)
- `diabetes, brings about, medical disorder` (vague)   
- `diabetes, brings about, adult` (incomplete)
- `diabetes, brings about, bone` (irrelevant) 

Output Format  
1. Analyze the triple against the criteria.  
2. Conclude with \boxed{True} (valid) or \boxed{False} (invalid).  
"""
#about 500 tokens in system_prompt and prompt

responses = []

# Process each query and save responses
for i, prompt in enumerate(prompts):
    prompt = prompt.strip()  # Remove leading/trailing whitespace
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    # Build the current batch
    if i % batch_size == 0:
        messages = [message]
    else:
        messages.append(message)

    # When the batch is complete or we're at the end of the dataset, process it.
    if i % batch_size == batch_size - 1 or i == len(prompts) - 1:
        logger.info(f"Processing batch to idx {i}:")
        outputs = llm.chat(messages, sampling_params=sampling_params, use_tqdm=False)
        for out in outputs:
            response = out.outputs[0].text
            responses.append(response)        

            
response_file = f"/scratch/gpfs/jx0800/outputs/triple_validation_result_llama8B_{task_id}.log"
with open(response_file, "w") as output_file:
    for i, prompt in enumerate(prompts):
        output_file.write(f"Query {i}: {prompt}\n")
        output_file.write(f"Response: {responses[i]}\n")
        output_file.write("-" * 50 + "\n")
