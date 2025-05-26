import json
import matplotlib.pyplot as plt
import numpy as np
with open('responses/qwen3-1.7b_gsm8k.json') as f:
    data = json.load(f)

lengths = [ex['thinking_length'] for ex in data]
filtered_lengths = [length for length in lengths if length != -1]

plt.hist(filtered_lengths, bins=30, alpha=0.7, edgecolor='black')
plt.xlabel("Thinking Length (tokens)")
plt.ylabel("Frequency")
plt.title("Distribution of Thinking Length in Model Responses")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("responses/qwen3-1.7b_gsm8k.png")

filtered_lengths = [length for length in lengths if length != -1]
tenth_percentile = np.percentile(filtered_lengths, 10)
ninetieth_percentile = np.percentile(filtered_lengths, 90)

print(f"10th percentile length: {tenth_percentile}")
print(f"90th percentile length: {ninetieth_percentile}")

