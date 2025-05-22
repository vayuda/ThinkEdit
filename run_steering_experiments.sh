#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# csv_file="qwen3-1.7b_steering_results.csv"
# == for mlp layers ==
# strengths=("-0.02" "0.02" "0.04" "0.06" "0.08")
# for strength in "${strengths[@]}"; do
#     echo "Running steering with strength: $strength"
#     torchrun --nproc_per_node=4 steer_thinking_fast.py \
#         --model qwen3-1.7b \
#         --direction_weight $strength \
#         --batch_size_per_gpu 32 \
#         --test_samples 1000 \
#         --max_new_tokens 4096 \
#         --output_dir steering_results_gsm8k
#     sleep 1
#     command=" python combine_steering_results.py \
#         --input_dir /home/pawan/code-bases/interp/ThinkEdit/steering_results_gsm8k \
#         --output_dir /home/pawan/code-bases/interp/ThinkEdit/steering_results_gsm8k \
#         --model qwen3-1.7b \
#         --control thinking_length_mlp \
#         --direction_weight $strength"
#     output=$(eval "$command")
#     accuracy=$(echo "$output" | grep "Accuracy:" | awk '{print $2}')
#     avg_thinking_length=$(echo "$output" | grep "Average thinking length:" | awk '{print $4}')
#     echo "$strength,$accuracy,$avg_thinking_length" >> "$csv_file"
#     sleep 1
# done

# == for attention layers ==
strengths=("-0.08" "-0.04" "-0.02")
for strength in "${strengths[@]}"; do
    echo "Running steering with strength: $strength"
    torchrun --nproc_per_node=8 steer_thinking_fast.py \
        --model qwen3-1.7b \
        --control thinking_length_attn \
        --direction_weight $strength \
        --batch_size_per_gpu 32 \
        --test_samples 1000 \
        --max_new_tokens 4096 \
        --output_dir steering_results_gsm8k
    sleep 1
    command=" python combine_steering_results.py \
        --input_dir /home/pawan/code-bases/interp/ThinkEdit/steering_results_gsm8k \
        --output_dir /home/pawan/code-bases/interp/ThinkEdit/steering_results_gsm8k \
        --model qwen3-1.7b \
        --control thinking_length_mlp \
        --direction_weight $strength"
    output=$(eval "$command")
    accuracy=$(echo "$output" | grep "Accuracy:" | awk '{print $2}')
    avg_thinking_length=$(echo "$output" | grep "Average thinking length:" | awk '{print $4}')
    echo "$strength,$accuracy,$avg_thinking_length" >> "$csv_file"
    sleep 1
done
