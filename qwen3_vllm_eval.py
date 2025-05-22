from vllm import LLM, SamplingParams
from datasets import load_dataset
from utils import get_think_length, extract_answer, model_dict
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from math_grader import math_equal, strip_string
import json
import re

def extract_ground_truth(answer_text):
    try:
        ret = int(answer_text)
    except ValueError:
        match = re.search(r'####\s*(\d+)', answer_text)
        if match:
            ret = int(match.group(1))
        else:
            ret = None
    finally:
        return ret

def build_vllm_prompts(dataset, tokenizer):
    prompts = []
    ids = []
    answers = []

    for id, item in enumerate(dataset):
        question = item['question']
        answer = item['answer']

        # Format into chat prompt string
        message = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  # if your tokenizer supports it
        )

        prompts.append(prompt)
        ids.append(id)
        answers.append(answer)

    return prompts, ids, answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3-1.7b", choices=["qwen3-1.7b", "deepseek-qwen-1.5b", "deepseek-llama3-8b", "deepseek-qwen-14b"])
    parser.add_argument("--control", type=str, default="thinking_length_mlp", choices=["thinking_length_mlp", "thinking_length_attn"])
    parser.add_argument("--direction_weight", type=float, default=0.00)
    parser.add_argument("--batch_size_per_gpu", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--output_dir", type=str, default="gsm8k_all_layer_thinking_length_steering_results", help="Directory to save results")
    parser.add_argument("--max_new_tokens", type=int, default=8192, help="Max new tokens for generation")
    parser.add_argument("--test_samples", type=int, default=200, help="Number of test samples to use")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="Dataset to use")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Model and Tokenizer ---
    model_path = model_dict[args.model]
    print(f"Loading model: {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    print(f"Loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    THINK_START_TOKEN_ID = tokenizer.encode("<think>", add_special_tokens=False)[0]
    THINK_END_TOKEN_ID = tokenizer.encode("</think>", add_special_tokens=False)[0]

    # --- Load Direction Vectors ---
    print(f"Loading direction vectors for {args.control}")
    
    if args.control == "thinking_length_mlp":
        direction = torch.load(f"directions/{args.model}_thinking_length_direction_gsm8k_mlp.pt").to(device)
    elif args.control == "thinking_length_attn":
        direction = torch.load(f"directions/{args.model}_thinking_length_direction_gsm8k_attn.pt").to(device)

    # --- Apply Hooks ---
    if "mlp" in args.control:
        def adjust_residual_hook(layer_idx):
            def hook_fn(module, input, output):
                return output + args.direction_weight * direction[layer_idx]
            return hook_fn

        print("Adding MLP hooks")
        for i, layer in enumerate(model.model.layers):
            layer.mlp.register_forward_hook(adjust_residual_hook(i))
    elif "attn" in args.control:
        def adjust_residual_hook(layer_idx):
            def hook_fn(module, input, output):
                return (output[0] + args.direction_weight * direction[layer_idx],) + output[1:]
            return hook_fn

        print("Adding attention hooks")
        for i, layer in enumerate(model.model.layers):
            layer.self_attn.register_forward_hook(adjust_residual_hook(i))


    # Load dataset and tokenizer
    if args.dataset == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test[:args.test_samples]")
    else:
        # load math
        dataset = load_dataset("csv", data_files="math_level5_140_test_examples.csv")

    # Build prompts
    prompts, ids, answers = build_vllm_prompts(dataset, tokenizer)

    # Load model
    llm = LLM(
        model=model_dict[args.model],
        tensor_parallel_size=4,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        swap_space=4
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.75,
        do_sample=True,
        max_new_tokens=args.max_new_tokens,
    )
    resampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.75,
        do_sample=True,
        max_new_tokens=256,
    )
    # Run generation
    outputs = llm.generate(prompts, sampling_params)

    all_responses = []
    all_think_lengths = []
    all_correctness = []
    all_correct_count = 0
    all_total_count = 0

    # Evaluate
    for id_, gt, out in zip(ids, answers, outputs):
        output_text = out.outputs[0].text.strip()
        output_ids = out.outputs[0].tokens
        
        # Extract thinking length
        think_length, complete_thinking = get_think_length(
            output_ids, THINK_START_TOKEN_ID, THINK_END_TOKEN_ID, max_length=args.max_new_tokens
        )
        if think_length == -1:
            continue

        # Handle case where thinking is too long and we need to generate final answer separately
        if think_length >= args.max_new_tokens:
            output_text += "\n</think>\n\nYeah, I think that's right.\n\n**Final Answer**\n"
            toks_continuation = tokenizer(output_text, return_tensors="pt").to(device)
            output = llm.generate(output_text, resampling_params)
            output_text = output[0].outputs[0].text.strip()

        # Extract and evaluate the answer
        predicted_answer = strip_string(extract_answer(output_text))
        ground_truth = extract_ground_truth(gt)
        
        all_total_count += 1
        is_correct = math_equal(predicted_answer, ground_truth)
        all_correct_count += int(is_correct)
        
        # Store the results
        all_responses.append({
            "id": id_,
            "prompt": prompts[i],
            "response": output_text,
            "predicted_answer": predicted_answer,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "thinking_length": think_length
        })
        
        all_think_lengths.append(think_length)
        


    # Continue with calculating and saving results
    accuracy = all_correct_count / all_total_count if all_total_count > 0 else 0

    # --- Calculate Results and Save ---
    accuracy = all_correct_count / all_total_count if all_total_count > 0 else 0
    avg_think_length = sum(all_think_lengths) / len(all_think_lengths) if all_think_lengths else 0

    print(f"Accuracy after times {args.direction_weight}: {accuracy:.4f}")
    print(f"Average thinking length: {avg_think_length:.2f}")

    # Prepare results object
    results = {
        "responses": all_responses,
        "think_lengths": all_think_lengths,
        "avg_thinking_length": avg_think_length,
        "correctness": all_correctness,
        "accuracy": accuracy,
        "direction_weight": args.direction_weight,
        "model": args.model,
        "control": args.control
    }

    # Create output directory structure
    output_path = os.path.join(args.output_dir, args.control, args.model)
    os.makedirs(output_path, exist_ok=True)

    # Save results
    with open(os.path.join(output_path, f"{args.direction_weight}.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Plot thinking length distribution
    plt.figure(figsize=(10, 6))
    bin_edges = list(range(0, 8500, 100)) + [8900, 9000]
    plt.hist(
        [t if t < args.max_new_tokens else 9000 for t in all_think_lengths], 
        bins=bin_edges, 
        alpha=0.7, 
        edgecolor='black', 
        label="Thinking Length"
    )
    plt.xticks(
        list(range(0, 9000, 1000)) + [9000], 
        labels=[str(i) for i in range(0, 9000, 1000)] + [f'>{args.max_new_tokens}']
    )
    plt.xlabel("Thinking Length (tokens)")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Thinking Length After Steering (times {args.direction_weight})")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_path, f"{args.direction_weight}.png"))
    print(f"Results saved to {output_path}")