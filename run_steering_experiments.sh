#!/bin/bash
export VLLM_ENABLE_V1_MULTIPROCESSING=0
csv_file="qwen3-1.7b_mlp-all_gsm8k_steering_results.csv"
# == for mlp layers ==
strengths=("-0.08" "-0.06" "-0.04" "-0.02" "0.00" "0.02" "0.04" "0.06" "0.08")
for strength in "${strengths[@]}"; do
    echo "Running steering with strength: $strength"
    command="python3 steering_eval.py \
        --model qwen3-1.7b \
        --weight $strength \
        --batch_size 64 \
        --n 500"
    output=$(eval "$command")
    accuracy=$(echo "$output" | grep "Accuracy:" | awk '{print $2}')
    avg_thinking_length=$(echo "$output" | grep "Average_thinking_length:" | awk '{print $2}')
    printf "%s,%s,%s\n" \
       "$strength" "$accuracy" "$avg_thinking_length" \
    >> "$csv_file"
done

csv_file="qwen3-1.7b_attn-all_gsm8k_steering_results.csv"
# == for attn layers ==
strengths=("-0.08" "-0.06" "-0.04" "-0.02" "0.00" "0.02" "0.04" "0.06" "0.08")
for strength in "${strengths[@]}"; do
    echo "Running steering with strength: $strength"
    command="python3 steering_eval.py \
        --model qwen3-1.7b \
        --weight $strength \
        --batch_size 64 \
        --n 500"
    output=$(eval "$command")
    accuracy=$(echo "$output" | grep "Accuracy:" | awk '{print $2}')
    avg_thinking_length=$(echo "$output" | grep "Average_thinking_length:" | awk '{print $2}')
    printf "%s,%s,%s\n" \
       "$strength" "$accuracy" "$avg_thinking_length" \
    >> "$csv_file"
done
