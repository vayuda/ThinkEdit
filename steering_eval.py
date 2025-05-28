import gc
import os
import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import pickle
import re
import matplotlib.pyplot as plt
import numpy as np
import json
from vllm import SamplingParams, LLM
from utils import get_think_length, extract_answer, model_dict, DATASET_MAP, extract_thinking
from math_grader import math_equal, strip_string
from transformers import AutoConfig
import itertools

np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="qwen3-1.7b", choices=["qwen3-1.7b","deepseek-qwen-1.5b", "deepseek-llama3-8b", "deepseek-qwen-14b"])
parser.add_argument("--control", type=str, default="mlp", choices=["mlp", "attn"])
parser.add_argument("--weight", type=float, default=0.00)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--dataset", type=str, choices=["gsm8k","gsm8k-e2h"], default="gsm8k-e2h")
parser.add_argument("--n", type=int, default=500, help = "Number of samples to evaluate use -1 for full dataset")
parser.add_argument("--tp", type=int, default=1)
args = parser.parse_args()

model_path = model_dict[args.model]
# model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device).eval()
model = LLM(model_path, tensor_parallel_size=args.tp, gpu_memory_utilization=0.9, enforce_eager=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
def get_thinking_text(response):
    return extract_thinking(response, tokenizer)
model_config = AutoConfig.from_pretrained(model_path)
THINK_START_TOKEN_ID = tokenizer.encode("<think>", add_special_tokens=False)[0]
THINK_END_TOKEN_ID = tokenizer.encode("</think>", add_special_tokens=False)[0]

dsinfo = DATASET_MAP[args.dataset]
qkey = dsinfo["question_key"]
akey = dsinfo["answer_key"]
ds_hf_path, ds_opts = dsinfo["args"]
dataset = load_dataset(ds_hf_path, ds_opts, split=dsinfo["split"])[:args.n]

if args.control == "mlp":
    direction = torch.load(f"directions/{args.model}_thinking_length_direction_gsm8k_mlp.pt").to(device)
elif args.control == "attn":
    direction = torch.load(f"directions/{args.model}_thinking_length_direction_gsm8k_attn.pt").to(device)


if "mlp" in args.control:
    def install_hooks(model):
        handlers = []
        for i in range(model.config.num_hidden_layers):
            def adjust_residual_hook():
                def hook_fn(module, input, out, idx=i):
                    return out + args.weight * direction[idx]
                return hook_fn
            handlers.append(model.model.layers[i].mlp.register_forward_hook(adjust_residual_hook()))
        return handlers
    handlers =  model.apply_model(install_hooks)
    print("add mlp hook")

elif "attn" in args.control:
    def install_hooks(model):
        handlers = []
        for i in range(model.config.num_hidden_layers):
            def adjust_residual_hook():
                def hook_fn(module, input, out, idx=i):
                    return out + args.weight * direction[idx]
                return hook_fn
            handlers.append(model.model.layers[i].self_attn.register_forward_hook(adjust_residual_hook()))
        return handlers
    handlers =  model.apply_model(install_hooks)
    print("add attn hook")

responses = []
think_lengths = []
correctness = []
correct_count = 0
total_count = 0

def extract_ground_truth(answer_text):
    match = re.search(r'####\s*(\d+)', answer_text)
    return int(match.group(1)) if match else None

def get_prompt(q, tokenizer):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": q}],
        tokenize=False,
        add_generation_prompt=True,
    )

def get_rerun_prompt(q, partial_response, tokenizer):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": q }, {"role": "assistant", "content": partial_response}],
        tokenize=False,
        add_generation_prompt=False,
    )



def batched(iterable, n):
    """Yield lists of length ≤ n from *iterable*."""
    it = iter(iterable)
    while (chunk := list(itertools.islice(it, n))):
        yield chunk

sp = SamplingParams(temperature=0.6, max_tokens=4096, top_p=0.95, n=1, best_of=1)
rerun_sp = SamplingParams(temperature=0.6, max_tokens=256, top_p=0.95, n=1, best_of=1)
for batch_rows in batched(dataset, args.batch_size):
    prompts = [get_prompt(r[qkey], tokenizer) for r in batch_rows]
    gens = model.generate(prompts, sp)
    rerun = []
    rerun = []
    for i, out in enumerate(gens):
        txt = out.outputs[0].text
        gens[i] = txt
        if len(txt) >= 4096 and "</think>" not in txt:
            rerun.append((i, txt))
    if rerun:
        new_prompts = [get_rerun_prompt(batch_rows[i][qkey], rerun[i][1], tokenizer) for i in range(len(rerun))]
        rerun_gens = model.generate(new_prompts, rerun_sp)
        for i, r in enumerate(rerun):
            gens[rerun[i][0]] = r[1] + rerun_gens[i].outputs[0].text

    for row, output in zip(batch_rows, gens):
        think_lengths.append(get_thinking_text(output)[1])
        predicted_answer = strip_string(extract_answer(output))
        responses.append(output)
        ground_truth = extract_ground_truth(row[akey])

        score = row["rating"] if args.dataset == "gsm8k-e2h" else 1
        total_count += score
        if math_equal(predicted_answer, ground_truth):
            correct_count += score
            correctness.append(1)
        else:
            correctness.append(0)

accuracy = correct_count / total_count if total_count > 0 else 0
print(f"Accuracy: {accuracy:.4f}")
print(f"Average_thinking_length: {sum(think_lengths) / len(think_lengths) if think_lengths else 0}")

results = {
    "responses": responses,
    "think_lengths": think_lengths,
    "avg_thinking_length": sum(think_lengths) / len(think_lengths),
    "correctness": correctness,
    "accuracy": accuracy
}
os.makedirs(f"{args.dataset}_all_layer_thinking_length_steering_results/{args.control}/{args.model}", exist_ok=True)
with open(f"{args.dataset}_all_layer_thinking_length_steering_results/{args.control}/{args.model}/{args.weight}.json", "w") as f:
    json.dump(results, f, indent=4)

# Plot thinking length distribution
plt.figure(figsize=(10, 6))
bin_edges = list(range(0, 8500, 100)) + [8900, 9000]
plt.hist([t if t < 8192 else 9000 for t in think_lengths], bins=bin_edges, alpha=0.7, edgecolor='black', label="Thinking Length")
plt.xticks(list(range(0, 9000, 1000)) + [9000], labels=[str(i) for i in range(0, 9000, 1000)] + ['>8192'])
plt.xlabel("Thinking Length (tokens)")
plt.ylabel("Frequency")
plt.title(f"Distribution of Thinking Length After Steering (times {args.weight})")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(f"{args.dataset}_all_layer_thinking_length_steering_results/{args.control}/{args.model}/{args.weight}.png")
