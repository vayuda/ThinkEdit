import json
import argparse

argparse = argparse.ArgumentParser()
argparse.add_argument("--model", type=str, default="qwen3-1.7b", choices=["deepseek-qwen-1.5b", "deepseek-llama3-8b", "deepseek-qwen-14b","qwen3-1.7b"])
args = argparse.parse_args()

json_file_path = f"responses/{args.model}_gsm8k.json"
with open(json_file_path, 'r') as f:
    responses_data = json.load(f)

# Filter examples based on thinking length
valid_responses = [ex for ex in responses_data if ex['thinking_length'] != -1]
long_thinking_examples = [ex for ex in valid_responses if ex['thinking_length'] > 1750]
short_thinking_examples = [ex for ex in valid_responses if ex['thinking_length'] < 500]

# change the key name from prompt to question
for example in long_thinking_examples:
    example['question'] = example.pop('prompt')

# save the modified data to a new JSON file
with open(f'{args.model}_gsm8k_long.json', 'w') as f:
    json.dump(long_thinking_examples, f, indent=4)

# change the key name from prompt to question
for example in short_thinking_examples:
    example['question'] = example.pop('prompt')

# save the modified data to a new JSON file
with open(f'{args.model}_gsm8k_short.json', 'w') as f:
    json.dump(short_thinking_examples, f, indent=4)
