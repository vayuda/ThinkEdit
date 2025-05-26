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
from utils import model_dict
from vllm import LLM, SamplingParams
from transformers.utils import logging
logging.set_verbosity_error()

import pickle
import time

np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="qwen3-1.7b", choices=["deepseek-qwen-1.5b", "deepseek-llama3-8b", "deepseek-qwen-14b","qwen3-1.7b"])
parser.add_argument("--vllm" , action="store_true")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--dataset", type=str, choices=["gsm8k"], default="gsm8k")
parser.add_argument("--ngpus", type=int, default=1)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

gsm8k = load_dataset('openai/gsm8k', 'main', split='train[:2000]')


model_path = model_dict[args.model]
if args.vllm:
    model = LLM(model_path, data_parallel_size=args.ngpus, gpu_memory_utilization=0.9)
    sp = SamplingParams(temperature=0.6, max_tokens=4096, top_p=0.95, n=1, best_of=1)
else: 
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device).eval()
    model.generation_config.do_sample = True

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
total = len(gsm8k['question'])

if args.vllm:
    for batch_rows in batched(gsm8k['question'], args.batch_size):
        prompts = [get_prompt(r, tokenizer) for r in batch_rows]

        gens = model.generate(prompts, sp)
        for q, out in zip(batch_rows, gens):
            output = out.outputs[0].text
        
            thinking_part, thinking_length = extract_thinking(output)
            thinking_lengths.append(thinking_length)
            responses_data.append({
                "question": q,
                "response": output,
                "thinking": thinking_part,
                "thinking_length": thinking_length
            })
else:
    for idx, q in enumerate(gsm8k['question']):
        print(f"Processing {idx+1}/{total}...", end='\r')
        toks = tokenizer(f"<｜User｜>{q}<｜Assistant｜>", return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(input_ids=toks['input_ids'].to(device), attention_mask=toks['attention_mask'].to(device), max_new_tokens=4096)[0]                        
        output = tokenizer.decode(output_ids[len(toks['input_ids'][0]):])
        print(output)
        output = output.replace("<\uff5cend\u2581of\u2581sentence\uff5c>", "")
        thinking_part, thinking_length = extract_thinking(output)
        thinking_lengths.append(thinking_length)

        # print(output)
        # print(thinking_length)
        
        responses_data.append({
            "question": q,
            "response": output,
            "thinking": thinking_part,
            "thinking_length": thinking_length
        })

os.makedirs("responses", exist_ok=True)
with open(f"responses/{args.model}_gsm8k.json", 'w') as f:
    json.dump(responses_data, f, indent=4)

# Plot thinking length distribution
plt.figure(figsize=(10, 6))
plt.hist(thinking_lengths, bins=30, alpha=0.7, edgecolor='black')
plt.xlabel("Thinking Length (tokens)")
plt.ylabel("Frequency")
plt.title("Distribution of Thinking Length in Model Responses")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(f"responses/{args.model}_thinking_length_distribution_gsm8k.png")
