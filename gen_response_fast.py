import os
import argparse
import json
import numpy as np
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import re
import functools
from utils import model_dict # Assuming utils.py contains your model_dict
from tqdm import tqdm
from transformers.utils import logging
logging.set_verbosity_error()

import time

# --- DDP Setup ---
def setup_ddp():
    """Initializes the distributed process group."""
    if not dist.is_available() or not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        print("DDP not available or not required (single GPU/CPU). Running in non-distributed mode.")
        return 0, 1 # rank 0, world_size 1

    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
         # Fallback for non-torchrun launch (e.g. debugging in an IDE) - treat as single process
        print("DDP environment variables not set. Running in non-distributed mode.")
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355' # Default port, ensure it's free
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        # Don't initialize process group here if only 1 device
        if torch.cuda.device_count() > 1:
             print("Warning: DDP environment variables not set, but multiple GPUs detected. Consider using torchrun.")
             # Decide if you want to force initialization or raise error
             # For now, continue as single process
             return 0, 1
        else:
            return 0, 1


    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ.get('LOCAL_RANK', rank % torch.cuda.device_count())) # Get local rank

    print(f"Initializing DDP: RANK={rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}")

    # Initializes the distributed backend NO INFORMATION IS SENT YET
    # nccl is generally recommended for NVIDIA GPUs
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

# --- Dataset and DataLoader ---
class GSM8KDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['question'])

    def __getitem__(self, idx):
        # Return the raw data, tokenization will happen in collate_fn
        return {"id": idx, "question": self.data['question'][idx]}

def collate_fn(batch, tokenizer, device, is_main_process=True):
    """Prepares a batch for the model."""
    questions = [item['question'] for item in batch]
    ids = [item['id'] for item in batch] # Keep track of original indices if needed

    # Format prompts
    messages = [[{"role": "user", "content": q}] for q in questions]
    texts = [tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
        ) for message in messages]

    # Tokenize the batch
    toks = tokenizer(texts, return_tensors="pt", padding=True, truncation=False) # Pad to longest in batch
    # Move tensors to the correct device
    toks = {k: v.to(device) for k, v in toks.items()}

    return {"ids": ids, "prompts": questions, "toks": toks}

# --- Helper Functions ---
def extract_thinking(response):
    """Extracts the thinking part from response text, including the <think> tags."""
    # Use re.IGNORECASE for case-insensitivity if needed
    match = re.search(r"(<think>.*?</think>)", response, re.DOTALL | re.IGNORECASE)
    if match:
        thinking_text = match.group(1).strip()
        # Calculate length *after* extraction to avoid tokenizing padding/extra chars
        # Note: This tokenizes again, could be optimized if performance critical
        # by analyzing the output_ids directly, but this is simpler.
        thinking_length = len(tokenizer(thinking_text, return_tensors='pt')['input_ids'][0])
        return thinking_text, thinking_length
    return "", -1 # Return -1 or 0 for length if no <think> tag is found

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3-1.7b", choices=["deepseek-qwen-1.5b", "deepseek-llama3-8b", "deepseek-qwen-14b","qwen3-1.7b"])
    parser.add_argument("--batch_size_per_gpu", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--output_dir", type=str, default="responses", help="Directory to save results")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Max new tokens for generation") # Reduced default from 4096 to save memory/time
    args = parser.parse_args()
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        # Might happen if context is already set, check if it's already 'spawn'
        if mp.get_start_method(allow_none=True) != 'spawn':
             print(f"Warning: Could not set start method to 'spawn': {e}")
        else:
             print("Start method already 'spawn'.")
    # --- DDP Initialization ---
    local_rank, world_size = setup_ddp()
    is_main_process = (local_rank == 0) # Check if this is the main process (rank 0)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() and world_size > 1 else \
             ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Rank {local_rank}] Using device: {device}")

    # Seed setting (important for reproducibility, especially with DDP)
    seed = 20 + local_rank # Offset seed by rank for potentially different initializations if needed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Load Model and Tokenizer ---
    # Ensure model loading happens after DDP setup, especially device placement
    model_path = model_dict[args.model]
    if is_main_process: print(f"Loading model: {model_path}")
    # Load model initially on CPU or meta device to avoid OOM on rank 0 if model is large, then move
    # Or load directly to the assigned GPU if memory allows
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 # or float16 if bf16 not supported
    ).to(device) # Move model to the assigned GPU *before* wrapping
    model.eval() # Set to evaluation mode

    if is_main_process: print("Attempting torch.compile()...")
    try:
        # Choose a mode. 'default' or 'reduce-overhead' often good for inference.
        # 'max-autotune' takes longer to compile but might yield best results.
        model = torch.compile(model, mode="reduce-overhead")
        if is_main_process: print("torch.compile() successful.")
    except Exception as e:
        if is_main_process: print(f"torch.compile() failed: {e}")

    if is_main_process: print(f"Loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    tokenizer.padding_side  = 'left'
    # Set padding token - important for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Configure generation - ensure pad_token_id is set
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id # Ensure EOS token is correctly set
    model.generation_config.do_sample = False # Usually False for tasks like GSM8K for reproducibility unless exploring solutions
    if is_main_process: print(f"Generation config: {model.generation_config}")

    # --- Wrap Model with DDP ---
    if world_size > 1:
        print(f"[Rank {local_rank}] Wrapping model with DDP...")
        # `find_unused_parameters=False` is often safe for inference or if all params are used
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        print(f"[Rank {local_rank}] Model wrapped.")
        # Use model.module to access original model methods after DDP wrapping
        model_to_generate = model.module
    else:
        model_to_generate = model # No wrapping needed

    # --- Load Dataset and Prepare DataLoader ---
    if is_main_process: print("Loading dataset...")
    # Load full dataset, sampler will distribute it
    full_gsm8k_dataset = load_dataset('openai/gsm8k', 'main', split='train[:2048]') # Load only once
    dataset = GSM8KDataset(full_gsm8k_dataset)

    # Create sampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=False) if world_size > 1 else None

    # Create DataLoader
    # Adjust num_workers based on your system
    collate_wrapper = functools.partial(collate_fn, tokenizer=tokenizer, device=device, is_main_process=is_main_process)

    # Create DataLoader using the wrapper
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size_per_gpu,
        sampler=sampler,
        collate_fn=collate_wrapper,
        shuffle=False if sampler else True, # Shuffle only if not using DistributedSampler
        num_workers=2,
        # pin_memory=True if torch.cuda.is_available() else False
    )

    if is_main_process: print(f"Dataset size: {len(dataset)}, Batches per epoch on rank 0: {len(dataloader)}")

    # --- Generation Loop ---
    local_responses_data = []
    local_thinking_lengths = []
    total_processed_on_rank = 0
    start_time = time.time()

    if world_size > 1 : dist.barrier() # Ensure all processes start generation roughly together
    for batch_idx, batch in enumerate(tqdm(dataloader, disable=not is_main_process)):
        if is_main_process:
            print(f"Processing batch {batch_idx + 1}/{len(dataloader)}...", end='\r')

        toks = batch['toks']
        prompts = batch['prompts'] # Keep original prompts for context if needed
        original_ids = batch['ids'] # Keep track of original dataset index

        with torch.no_grad():
            # Generate responses for the batch
            # Make sure max_length accounts for prompt length + max_new_tokens
            # Using max_new_tokens is generally preferred
            output_ids = model_to_generate.generate(
                input_ids=toks['input_ids'],
                attention_mask=toks['attention_mask'],
                max_new_tokens=args.max_new_tokens,
                generation_config=model_to_generate.generation_config # Pass the config explicitly
            ) # output_ids shape: (batch_size, seq_len)

        # Decode and process each response in the batch
        # output_ids contains the prompt + generated text
        # We need to slice off the prompt part for decoding the *new* tokens only
        prompt_lengths = [len(toks['input_ids'][i]) for i in range(len(prompts))]
        generated_sequences = tokenizer.batch_decode(output_ids, skip_special_tokens=True) # Decode full sequences first
        
        # Process results from the batch
        for i in range(len(prompts)):
            # Get only the generated part by slicing based on prompt token length
            # Need to re-tokenize the prompt to get its length *reliably* if padding occurred
            # Or, slice the output_ids tensor before decoding:
            response_ids = output_ids[i][prompt_lengths[i]:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
            
            # Clean potential special tokens if not skipped properly (like the odd one you saw)
            response_text = response_text.replace("<\uff5cend\u2581of\u2581sentence\uff5c>", "").strip() # Example cleaning

            thinking_part, thinking_length = extract_thinking(response_text)

            # Find the original question using the ID if needed (less efficient but robust)
            # original_question = full_gsm8k_dataset[original_ids[i]]['question'] # Example lookup

            local_responses_data.append({
                "id": original_ids[i], # Store original index
                # "question": original_question, # Store question if needed
                "prompt": prompts[i], # Store the actual prompt sent to model
                "response": response_text,
                "thinking": thinking_part,
                "thinking_length": thinking_length
            })
            if thinking_length != -1: # Only append valid lengths for histogram
                 local_thinking_lengths.append(thinking_length)

        total_processed_on_rank += len(prompts)
        
        # Optional: Clear cache periodically if memory is tight
        # if batch_idx % 10 == 0:
        #     gc.collect()
        #     if torch.cuda.is_available():
        #         torch.cuda.empty_cache()

    # --- Gather Results (if DDP is enabled) ---
    all_responses_data = []
    all_thinking_lengths = []

    if world_size > 1:
        print(f"\n[Rank {local_rank}] Finished generation. Processed {total_processed_on_rank} samples. Gathering results...")
        dist.barrier() # Wait for all processes to finish generation

        # Gather lists of dictionaries (requires pickle)
        gathered_responses = [None] * world_size
        dist.all_gather_object(gathered_responses, local_responses_data) # Gathers objects from all processes into a list on each process

        gathered_lengths = [None] * world_size
        dist.all_gather_object(gathered_lengths, local_thinking_lengths) # Gather lengths similarly

        if is_main_process:
            print("Gathering complete. Combining results...")
            # Combine results on the main process
            for data_list in gathered_responses:
                all_responses_data.extend(data_list)
            for length_list in gathered_lengths:
                all_thinking_lengths.extend(length_list)
            # Optional: Sort results by original ID if order matters
            all_responses_data.sort(key=lambda x: x['id'])
            print(f"Total responses collected: {len(all_responses_data)}")
            print(f"Total valid thinking lengths collected: {len(all_thinking_lengths)}")

    else: # Single process case
        all_responses_data = local_responses_data
        all_thinking_lengths = local_thinking_lengths
        if is_main_process:
             print(f"\nFinished generation. Processed {total_processed_on_rank} samples.")

    # --- Save Results and Plot (only on main process) ---
    if is_main_process:
        end_time = time.time()
        print(f"Total processing time: {end_time - start_time:.2f} seconds")

        os.makedirs(args.output_dir, exist_ok=True)
        output_filename = os.path.join(args.output_dir, f"{args.model}_gsm8k.json")
        print(f"Saving responses to {output_filename}...")
        with open(output_filename, 'w') as f:
            json.dump(all_responses_data, f, indent=4)
        print("Responses saved.")

        # Plot thinking length distribution
        if all_thinking_lengths:
            plot_filename = os.path.join(args.output_dir, f"{args.model}_thinking_length_dist_gsm8k_batched_ddp.png")
            print(f"Saving plot to {plot_filename}...")
            plt.figure(figsize=(10, 6))
            plt.hist(all_thinking_lengths, bins=30, alpha=0.7, edgecolor='black')
            plt.xlabel("Thinking Length (tokens)")
            plt.ylabel("Frequency")
            plt.title(f"Distribution of Thinking Length ({args.model} on GSM8K)")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(plot_filename)
            print("Plot saved.")
        else:
            print("No valid thinking lengths found to plot.")

    # --- DDP Cleanup ---
    cleanup_ddp()
    print(f"[Rank {local_rank}] Script finished.")