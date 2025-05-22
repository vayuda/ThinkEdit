import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
import matplotlib.pyplot as plt
import numpy as np
import json
import functools
from utils import get_think_length, extract_answer, model_dict
from math_grader import math_equal, strip_string
from tqdm import tqdm

# --- DDP Setup ---
def setup_ddp():
    """Initializes the distributed process group."""
    if not dist.is_available() or not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        print("DDP not available or not required (single GPU/CPU). Running in non-distributed mode.")
        return 0, 1  # rank 0, world_size 1

    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        # Fallback for non-torchrun launch (e.g. debugging in an IDE) - treat as single process
        print("DDP environment variables not set. Running in non-distributed mode.")
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'  # Default port, ensure it's free
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        if torch.cuda.device_count() > 1:
            print("Warning: DDP environment variables not set, but multiple GPUs detected. Consider using torchrun.")
            return 0, 1
        else:
            return 0, 1

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ.get('LOCAL_RANK', rank % torch.cuda.device_count()))  # Get local rank

    print(f"Initializing DDP: RANK={rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}")

    # Initializes the distributed backend
    dist.init_process_group(backend='nccl', init_method='env://')

    # Set the device for the current process
    torch.cuda.set_device(local_rank)
    print(f"[Rank {rank}] Using GPU: {torch.cuda.current_device()}")

    return local_rank, world_size

def cleanup_ddp():
    """Cleans up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("DDP process group destroyed.")

# --- Dataset Class ---
class GSM8KDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['question'])

    def __getitem__(self, idx):
        return {
            "id": idx,
            "question": self.data['question'][idx],
            "answer": self.data['answer'][idx]
        }

class MathDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data['question'])

    def __getitem__(self, idx):
        return {
            "id": idx,
            "question": self.data['problem'][idx],
            "answer": self.data['answer'][idx]
        }

# --- Collate Function ---
def collate_fn(batch, tokenizer, device):
    """Prepares a batch for the model."""
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]
    ids = [item['id'] for item in batch]

    # Format prompts
    messages = [[{"role": "user", "content": q}] for q in questions]
    texts = [tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    ) for message in messages]

    # Tokenize the batch
    toks = tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
    # Move tensors to the correct device
    toks = {k: v.to(device) for k, v in toks.items()}

    return {"ids": ids, "prompts": questions, "answers": answers, "toks": toks}

def extract_ground_truth(answer_text):
    match = re.search(r'####\s*(\d+)', answer_text)
    return int(match.group(1)) if match else None

# --- Main Execution ---
if __name__ == "__main__":
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

    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        if mp.get_start_method(allow_none=True) != 'spawn':
            print(f"Warning: Could not set start method to 'spawn': {e}")
        else:
            print("Start method already 'spawn'.")

    # --- DDP Initialization ---
    local_rank, world_size = setup_ddp()
    is_main_process = (local_rank == 0)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() and world_size > 1 else \
             ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Rank {local_rank}] Using device: {device}")

    # Seed setting for reproducibility
    seed = 20 + local_rank
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Load Model and Tokenizer ---
    model_path = model_dict[args.model]
    if is_main_process:
        print(f"Loading model: {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    if is_main_process:
        print(f"Loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    THINK_START_TOKEN_ID = tokenizer.encode("<think>", add_special_tokens=False)[0]
    THINK_END_TOKEN_ID = tokenizer.encode("</think>", add_special_tokens=False)[0]

    # --- Load Direction Vectors ---
    if is_main_process:
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

        if is_main_process:
            print("Adding MLP hooks")
        for i, layer in enumerate(model.model.layers):
            layer.mlp.register_forward_hook(adjust_residual_hook(i))
    elif "attn" in args.control:
        def adjust_residual_hook(layer_idx):
            def hook_fn(module, input, output):
                return (output[0] + args.direction_weight * direction[layer_idx],) + output[1:]
            return hook_fn

        if is_main_process:
            print("Adding attention hooks")
        for i, layer in enumerate(model.model.layers):
            layer.self_attn.register_forward_hook(adjust_residual_hook(i))

    # --- Wrap Model with DDP ---
    if world_size > 1:
        print(f"[Rank {local_rank}] Wrapping model with DDP...")
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        print(f"[Rank {local_rank}] Model wrapped.")
        model_to_generate = model.module
    else:
        model_to_generate = model  # No wrapping needed

    # --- Load Dataset and Prepare DataLoader ---
    if is_main_process:
        print("Loading dataset...")
    
    # Load only the specified number of test samples
    if args.dataset == "gsm8k":
        gsm8k_data = load_dataset('openai/gsm8k', 'main', split=f'test[:{args.test_samples}]')
        dataset = GSM8KDataset(gsm8k_data)
    else:
        math = load_dataset("csv", data_files="math_level5_140_test_examples.csv")
        dataset = MathDataset(math)

    torch.distributed.barrier()  # Ensure all processes have loaded the dataset
    # Create sampler for distributed execution
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=False) if world_size > 1 else None

    # Create DataLoader with the collate function
    collate_wrapper = functools.partial(collate_fn, tokenizer=tokenizer, device=device)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size_per_gpu,
        sampler=sampler,
        collate_fn=collate_wrapper,
        shuffle=False if sampler else True,
        num_workers=2
    )

    if is_main_process:
        print(f"Dataset size: {len(dataset)}, Batches per epoch on rank {local_rank}: {len(dataloader)}")

    # --- Generation Loop ---
    local_responses = []
    local_think_lengths = []
    local_correctness = []
    local_correct_count = 0
    local_total_count = 0

    if world_size > 1:
        dist.barrier()  # Ensure all processes start generation together
        
    for batch_idx, batch in enumerate(tqdm(dataloader, disable=not is_main_process)):

        toks = batch['toks']
        prompts = batch['prompts']
        answers = batch['answers']
        original_ids = batch['ids']

        with torch.no_grad():
            output_ids = model_to_generate.generate(
                input_ids=toks['input_ids'],
                attention_mask=toks['attention_mask'],
                max_new_tokens=args.max_new_tokens,
                do_sample=True
            )

        # Process each item in the batch
        for i in range(len(prompts)):
            # Get the complete output text
            output_text = tokenizer.decode(output_ids[i])
            
            # Extract thinking length
            think_length, complete_thinking = get_think_length(
                output_ids[i], THINK_START_TOKEN_ID, THINK_END_TOKEN_ID, max_length=args.max_new_tokens
            )
            # print(f"Sample {original_ids[i]}, Thinking length: {think_length}")
            
            # Skip if no thinking tags found
            if think_length == -1:
                continue

            # Handle case where thinking is too long and we need to generate final answer separately
            if think_length >= args.max_new_tokens:
                output_text += "\n</think>\n\nYeah, I think that's right.\n\n**Final Answer**\n"
                toks_continuation = tokenizer(output_text, return_tensors="pt").to(device)
                with torch.no_grad():
                    continuation_ids = model_to_generate.generate(
                        input_ids=toks_continuation['input_ids'],
                        attention_mask=toks_continuation['attention_mask'],
                        max_new_tokens=256,  # Generate only the final answer
                        do_sample=True
                    )
                output_text = tokenizer.decode(continuation_ids[0])

            # Extract and evaluate the answer
            predicted_answer = strip_string(extract_answer(output_text))
            ground_truth = extract_ground_truth(answers[i])
            
            local_total_count += 1
            is_correct = math_equal(predicted_answer, ground_truth)
            if is_correct:
                local_correct_count += 1
                # print(f"Sample {original_ids[i]}: Correct")
                local_correctness.append(1)
            else:
                # print(f"Sample {original_ids[i]}: Incorrect")
                local_correctness.append(0)
            
            # Store the results
            local_responses.append({
                "id": original_ids[i],
                "prompt": prompts[i],
                "response": output_text,
                "predicted_answer": predicted_answer,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "thinking_length": think_length
            })
            
            local_think_lengths.append(think_length)

    if world_size > 1:
        print(f"\n[Rank {local_rank}] Finished generation. Saving local results...")
        
        # Create output directory structure
        output_path = os.path.join(args.output_dir, args.control, args.model)
        os.makedirs(output_path, exist_ok=True)
        
        # Each process saves its own results
        rank_output_file = os.path.join(output_path, f"rank_{local_rank}_results_{args.direction_weight}.json")
        with open(rank_output_file, "w") as f:
            json.dump({
                "responses": local_responses,
                "think_lengths": local_think_lengths,
                "correctness": local_correctness,
                "correct_count": local_correct_count,
                "total_count": local_total_count,
                "ids": [r["id"] for r in local_responses]
            }, f)
        
        print(f"[Rank {local_rank}] Local results saved to {rank_output_file}")
        
        # No need for barrier or gathering - each process finishes independently
        print(f"[Rank {local_rank}] Script finished.")
    else:
        # Single process mode - process the results directly
        all_responses = local_responses
        all_think_lengths = local_think_lengths
        all_correctness = local_correctness
        all_correct_count = local_correct_count
        all_total_count = local_total_count
        
        # Continue with calculating and saving results
        accuracy = all_correct_count / all_total_count if all_total_count > 0 else 0

        # --- Calculate Results and Save (only on main process) ---
        if is_main_process:
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

    # --- DDP Cleanup ---
    cleanup_ddp()
    print(f"[Rank {local_rank}] Script finished.")