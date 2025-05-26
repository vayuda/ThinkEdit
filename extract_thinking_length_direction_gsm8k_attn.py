import gc
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from utils import model_dict

from transformers.utils import logging
logging.set_verbosity_error()
from tqdm import tqdm 
import pickle
import time
import json
import math
import re

np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="deepseek_llama3")
parser.add_argument("--hook_point", type=str, default="attn", choices=["attn", "mlp"])
args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"


# -- load model and tokenizer --
model_path = model_dict[args.model]
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
model.generation_config.do_sample = True
tokenizer.pad_token = tokenizer.eos_token




def extract_tl_dir(examples):
    # -- attach attn hook to retrieve residuals --
    if args.hook_point == "attn":
        print("attaching attn hook")
        attn_outputs = []

        def capture_residual_hook():
            def hook_fn(module, input, output):
                attn_outputs.append(input[0].detach())
            return hook_fn
        
        for layer in model.model.layers:
            if args.hook_point == "attn":
                layer.post_attention_layernorm.register_forward_hook(capture_residual_hook())
    embeddings = []
    for example in tqdm(examples):
        message = [{"role": "user", "content": example['question']}, {"role": "assistant", "content": ""}]
        question = tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
        )
        start = len(tokenizer(question).input_ids)

        message = [{"role": "user", "content": example['question']}, {"role": "assistant", "content": example['thinking']}]
        cot = tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
        )
        end = len(tokenizer(cot).input_ids)
        toks = tokenizer(cot, return_tensors="pt")

        # toks = tokenizer(f"<｜User｜>{example['question']}<｜Assistant｜>").input_ids
        # start = len(toks)
        # toks = tokenizer(f"<｜User｜>{example['question']}<｜Assistant｜>{example['thinking']}").input_ids
        # end = len(toks)
        # toks = tokenizer(f"<｜User｜>{example['question']}<｜Assistant｜>{example['thinking']}", return_tensors="pt")
        with torch.no_grad():
            if args.hook_point == "attn":
                _ = model(input_ids=toks['input_ids'].to(device), attention_mask=toks['attention_mask'].to(device))
                embeddings.append(torch.stack(attn_outputs, dim=0)[:, :, start-1:end-1, :].mean(dim=2).cpu())
                attn_outputs = []
            elif args.hook_point == "mlp":
                residual_outputs = model(input_ids=toks['input_ids'].to(device), attention_mask=toks['attention_mask'].to(device), output_hidden_states=True).hidden_states[1:]
                embeddings.append(torch.stack(residual_outputs, dim=0)[:, :, start-1:end-1, :].mean(dim=2).cpu())
    return torch.stack(embeddings, dim=0).mean(dim=0)

# Load JSON file with response data
json_file_path = f"responses/{args.model}_gsm8k.json"
with open(json_file_path, 'r') as f:
    responses_data = json.load(f)

# Filter examples based on thinking length


valid_responses = [ex for ex in responses_data if ex['thinking_length'] != -1]
tenth_percentile = np.percentile(valid_responses, 10)
ninetieth_percentile = np.percentile(valid_responses, 90)

long_thinking_examples = [ex for ex in valid_responses if ex['thinking_length'] > ninetieth_percentile]
short_thinking_examples = [ex for ex in valid_responses if ex['thinking_length'] < tenth_percentile]

# -- long examples --
print("number of long examples: ",len(long_thinking_examples))
mean_embedding_long = extract_tl_dir(long_thinking_examples)

# -- short examples --
print("number of short examples: ",len(short_thinking_examples))
mean_embedding_short = extract_tl_dir(short_thinking_examples)

# -- save embeddings --
thinking_length_direction = mean_embedding_long - mean_embedding_short
os.makedirs("directions", exist_ok=True)
torch.save(thinking_length_direction, f"directions/{args.model}_thinking_length_direction_gsm8k_{args.hook_point}.pt")
