#!/usr/bin/env python3
"""
plot_all_metrics.py

For each CSV in a given directory (with columns:
  steering_strength, accuracy, thinking_length)
this script generates two side-by-side plots:
  1) Accuracy vs. Steering Strength
  2) Thinking Length vs. Steering Strength
with dashed horizontal lines for each metric’s mean,
and saves each figure as <csv_basename>_metrics.png
in the same directory.
"""

import argparse
import glob
import os

import pandas as pd
import matplotlib.pyplot as plt

def plot_and_save(df, out_path, title):
    x = df['steering_strength']
    acc = df['accuracy']
    length = df['thinking_length']

    mean_acc = acc.mean()
    mean_len = length.mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    ax1.plot(x, acc, marker='o')
    ax1.hlines(mean_acc, x.min(), x.max(), linestyles='--')
    ax1.set_title(title)
    ax1.set_xlabel('Steering Strength')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True)

    # Thinking Length plot
    ax2.plot(x, length, marker='o')
    ax2.hlines(mean_len, x.min(), x.max(), linestyles='--')
    ax2.set_title(title)
    ax2.set_xlabel('Steering Strength')
    ax2.set_ylabel('Thinking Length')
    ax2.grid(True)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot and save metrics for all CSVs in a directory."
    )
    parser.add_argument(
        'dir_path',
        help="Path to directory containing CSV files"
    )
    args = parser.parse_args()
    dir_path = args.dir_path

    # Find all CSV files
    pattern = os.path.join(dir_path, '*.csv')
    csv_files = glob.glob(pattern)
    if not csv_files:
        parser.error(f"No CSV files found in {dir_path}")

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  Skipping {csv_path}: cannot read CSV ({e})")
            continue
        parts = os.path.basename(csv_path).split('_')
        dataset = parts[2]
        model = parts[0]
        control = parts[1]
        title = f"steering results with {model} intervening the {control} layers on {dataset}"
        required = {'steering_strength', 'accuracy', 'thinking_length'}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            print(f"  Skipping {csv_path}: missing columns {missing}")
            continue

        base = os.path.splitext(os.path.basename(csv_path))[0]
        out_png = os.path.join(dir_path, f"{base}_metrics.png")
        print(f"  Processing {csv_path} → {out_png}")
        plot_and_save(df, out_png, title)

    print("Done.")

if __name__ == "__main__":
    main()
