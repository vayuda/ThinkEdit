import gc
import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import re
from utils import model_dict, DATASET_MAP
from vllm import LLM, SamplingParams
from transformers.utils import logging
logging.set_verbosity_error()
import itertools
import pickle
import time

np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="qwen3-1.7b", choices=["deepseek-qwen-1.5b", "deepseek-llama3-8b", "deepseek-qwen-14b","qwen3-1.7b"])
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--dataset", type=str, choices=["gsm8k","gsm8k-e2h"], default="gsm8k-e2h")
parser.add_argument("--tp", type=int, default=1)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

dsinfo = DATASET_MAP[args.dataset]
ds_hf_path, ds_opts = dsinfo["args"]
dataset = load_dataset(ds_hf_path, ds_opts, split=dsinfo["split"])


model_path = model_dict[args.model]

sp = SamplingParams(temperature=0.6, max_tokens=8192, top_p=0.95, top_k=20)
model = LLM(model_path, tensor_parallel_size=args.tp, gpu_memory_utilization=0.95)


tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

def extract_thinking(response):
    """Extracts the thinking part from response text, including the <think> tags."""
    match = re.search(r"(<think>.*?</think>)", response, re.DOTALL)
    if match:
        thinking_text = match.group(1).strip()
        return thinking_text, len(tokenizer(thinking_text, return_tensors='pt')['input_ids'][0])
    return "", -1

def batched(iterable, n):
    """Yield lists of length ≤ n from *iterable*."""
    it = iter(iterable)
    while (chunk := list(itertools.islice(it, n))):
        yield chunk

def get_prompt(q, tokenizer):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": q}],
        tokenize=False,
        add_generation_prompt=True,
    )

responses_data = []
thinking_lengths = []
question_difficulties=[]
total = len(dataset)


for batch_rows in batched(dataset, args.batch_size):
    prompts = [get_prompt(r[dsinfo["question_key"]], tokenizer) for r in batch_rows]

    gens = model.generate(prompts, sp)
    for q, out in zip(batch_rows, gens):
        output = out.outputs[0].text
    
        thinking_part, thinking_length = extract_thinking(output)
        thinking_lengths.append(thinking_length)
        question_difficulty = q["rating"] if args.dataset == "gsm8k-e2h" else 1
        question_difficulties.append(question_difficulty)
        responses_data.append({
            "question": q,
            "question_difficulty": question_difficulty,
            "response": output,
            "thinking": thinking_part,
            "thinking_length": thinking_length
        })

os.makedirs("responses", exist_ok=True)
with open(f"responses/{args.model}_{args.dataset}.json", 'w') as f:
    json.dump(responses_data, f, indent=4)

# Plot thinking length distribution
plt.figure(figsize=(10, 6))
plt.hist(thinking_lengths, bins=30, alpha=0.7, edgecolor='black')
plt.xlabel("Thinking Length (tokens)")
plt.ylabel("Frequency")
plt.title("Distribution of Thinking Length in Model Responses")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(f"responses/{args.model}_thinking_length_distribution_{args.dataset}.png")

# plot coorelation between thinking length and question difficulty
plt.figure(figsize=(10, 6))
plt.scatter(question_difficulties, thinking_lengths, alpha=0.7)
plt.xlabel("Question Difficulty")
plt.ylabel("Thinking Length (tokens)")
plt.title("Thinking Length vs. Question Difficulty")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(f"responses/{args.model}_thinking_length_vs_question_difficulty_{args.dataset}.png")
