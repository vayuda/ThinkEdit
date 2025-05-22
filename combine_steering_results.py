#!/usr/bin/env python3
import os
import json
import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def combine_results(input_dir, direction_weight, model, control):
    """Combine results from multiple rank files into a single result."""
    # Find all rank result files for this configuration
    pattern = os.path.join(input_dir, control, model, f"rank_*_results_{direction_weight}.json")
    rank_files = glob.glob(pattern)
    
    if not rank_files:
        print(f"No result files found matching pattern: {pattern}")
        return None
    
    print(f"Found {len(rank_files)} rank result files")
    
    # Initialize combined data structures
    all_responses = []
    all_think_lengths = []
    all_correctness = []
    all_correct_count = 0
    all_total_count = 0
    
    # Process each rank file
    for rank_file in rank_files:
        with open(rank_file, 'r') as f:
            rank_data = json.load(f)
        
        all_responses.extend(rank_data['responses'])
        all_think_lengths.extend(rank_data['think_lengths'])
        all_correctness.extend(rank_data['correctness'])
        all_correct_count += rank_data['correct_count']
        all_total_count += rank_data['total_count']
    
    # Sort by original ID to maintain consistent order
    all_responses.sort(key=lambda x: x['id'])
    
    # Calculate results
    accuracy = all_correct_count / all_total_count if all_total_count > 0 else 0
    avg_think_length = sum(all_think_lengths) / len(all_think_lengths) if all_think_lengths else 0
    
    print(f"Combined results:")
    print(f"  Total samples: {all_total_count}")
    print(f"  Correct answers: {all_correct_count}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Average thinking length: {avg_think_length:.2f}")
    
    # Prepare results object
    results = {
        "responses": all_responses,
        "think_lengths": all_think_lengths,
        "avg_thinking_length": avg_think_length,
        "correctness": all_correctness,
        "accuracy": accuracy,
        "direction_weight": direction_weight,
        "model": model,
        "control": control
    }
    
    return results

def save_combined_results(results, output_dir, max_new_tokens):
    """Save the combined results and generate plots."""
    if not results:
        return
    
    model = results["model"]
    control = results["control"]
    direction_weight = results["direction_weight"]
    
    # Create output directory
    output_path = os.path.join(output_dir, control, model)
    os.makedirs(output_path, exist_ok=True)
    
    # Save combined results
    with open(os.path.join(output_path, f"{direction_weight}.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    # Plot thinking length distribution
    plt.figure(figsize=(10, 6))
    all_think_lengths = results["think_lengths"]
    bin_edges = list(range(0, max_new_tokens + 500, 100)) + [max_new_tokens + 500]
    plt.hist(
        [t if t < max_new_tokens else max_new_tokens + 500 for t in all_think_lengths], 
        bins=bin_edges, 
        alpha=0.7, 
        edgecolor='black', 
        label="Thinking Length"
    )
    plt.xticks(
        list(range(0, max_new_tokens + 500, 1000)) + [max_new_tokens + 500], 
        labels=[str(i) for i in range(0, max_new_tokens + 500, 1000)] + [f'>{max_new_tokens}']
    )
    plt.xlabel("Thinking Length (tokens)")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Thinking Length After Steering (times {direction_weight})")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_path, f"{direction_weight}.png"))
    
    print(f"Combined results saved to {output_path}")
    
    # Plot correctness vs thinking length
    plt.figure(figsize=(10, 6))
    
    # Group thinking lengths into bins and calculate accuracy per bin
    bin_size = 500
    bins = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for length, correct in zip(all_think_lengths, results["correctness"]):
        bin_idx = length // bin_size
        bins[bin_idx]["total"] += 1
        bins[bin_idx]["correct"] += correct
    
    # Prepare data for plotting
    bin_centers = []
    accuracies = []
    
    for bin_idx in sorted(bins.keys()):
        bin_data = bins[bin_idx]
        if bin_data["total"] >= 5:  # Only include bins with sufficient samples
            bin_centers.append((bin_idx * bin_size) + (bin_size // 2))
            accuracies.append(bin_data["correct"] / bin_data["total"])
    
    plt.plot(bin_centers, accuracies, 'o-', linewidth=2, markersize=8)
    plt.xlabel("Thinking Length (tokens)")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs Thinking Length (direction_weight={direction_weight})")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_path, f"{direction_weight}_accuracy_vs_length.png"))

def parse_args():
    parser = argparse.ArgumentParser(description="Combine distributed evaluation results")
    parser.add_argument("--input_dir", type=str, default="gsm8k_all_layer_thinking_length_steering_results",
                        help="Directory containing the rank result files")
    parser.add_argument("--output_dir", type=str, default="gsm8k_all_layer_thinking_length_steering_results",
                        help="Directory to save combined results")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name")
    parser.add_argument("--control", type=str, required=True,
                        help="Control type")
    parser.add_argument("--direction_weight", type=float, required=True,
                        help="Direction weight value")
    parser.add_argument("--max_new_tokens", type=int, default=8192,
                        help="Maximum new tokens in generation")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Combine results from all ranks
    results = combine_results(
        args.input_dir, 
        args.direction_weight,
        args.model,
        args.control
    )
    
    # Save combined results and generate plots
    if results:
        save_combined_results(results, args.output_dir, args.max_new_tokens)
    else:
        print("No results to combine.")