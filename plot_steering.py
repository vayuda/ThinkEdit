#!/usr/bin/env python3
"""
plot_steering_metrics_combined.py
================================

Aggregate steering‑metrics CSV files in a directory and create **two** summary
figures:

1. **Thinking Length vs. Steering Strength** – one line per CSV.
2. **Accuracy vs. Steering Strength**        – one line per CSV.

Each CSV must contain the columns:
    - steering_strength
    - accuracy
    - thinking_length

The CSV filename is expected to follow the pattern
    <model>_<control>_<dataset>_*.csv
so that a readable legend label can be derived automatically.  Files that do
not satisfy the column requirements are skipped (with a warning).  All final
PNGs are written into the same directory that holds the CSVs.

Example
-------
$ python plot_steering_metrics_combined.py ./results/gsm8k

This will create the files
    combined_thinking_length_vs_steering_strength.png
    combined_accuracy_vs_steering_strength.png
inside `./results/gsm8k`.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

REQUIRED_COLUMNS = {"steering_strength", "accuracy", "thinking_length"}

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _find_csv_files(dir_path: Path) -> List[Path]:
    """Return a list of *.csv files in *dir_path* (non‑recursive)."""
    return sorted(dir_path.glob("*.csv"))


def _read_metrics(csv_path: Path) -> Tuple[pd.DataFrame, str]:
    """Read *csv_path* and return *(df, legend_label)*.

    The legend label is derived from the first three underscore‑separated parts
    of the filename (model, control, dataset).
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001 – show the underlying problem
        raise RuntimeError(f"Cannot read {csv_path}: {exc}") from exc

    if not REQUIRED_COLUMNS.issubset(df.columns):
        missing = REQUIRED_COLUMNS.difference(df.columns)
        raise ValueError(f"Missing columns {missing} in {csv_path}")

    # Ensure proper dtype ordering (in case the CSV uses strings for strengths)
    df = df.copy()
    df["steering_strength"] = pd.to_numeric(df["steering_strength"], errors="coerce")
    df.sort_values("steering_strength", inplace=True)

    parts = csv_path.stem.split("_")
    label = " ".join(parts[:3]) if len(parts) >= 3 else csv_path.stem
    return df, label


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _setup_ax(ax: plt.Axes, x_label: str, y_label: str, title: str) -> None:
    """Apply shared axis cosmetics."""
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.6)


def _plot_metric(ax: plt.Axes, df: pd.DataFrame, x: str, y: str, label: str) -> None:
    """Plot *y* vs *x* on *ax* with markers and an automatic line colour."""
    ax.plot(df[x], df[y], marker="o", label=label)


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def build_combined_plots(dir_path: Path) -> None:
    """Create the two summary plots for all CSVs in *dir_path*."""
    csv_paths = _find_csv_files(dir_path)
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {dir_path}")

    # Prepare figures
    fig_len, ax_len = plt.subplots(figsize=(7, 5))
    fig_acc, ax_acc = plt.subplots(figsize=(7, 5))

    # Process each CSV
    skipped: List[str] = []
    for csv_path in csv_paths:
        try:
            df, label = _read_metrics(csv_path)
        except (RuntimeError, ValueError) as err:
            skipped.append(f"  Skipping {csv_path.name}: {err}")
            continue

        _plot_metric(ax_len, df, "steering_strength", "thinking_length", label)
        _plot_metric(ax_acc, df, "steering_strength", "accuracy", label)

    if skipped:
        print("\n".join(skipped))

    # Finalise figures
    _setup_ax(ax_len, "Steering Strength", "Thinking Length", "Thinking Length vs. Steering Strength")
    _setup_ax(ax_acc, "Steering Strength", "Accuracy", "Accuracy vs. Steering Strength")

    ax_len.legend(fontsize="small", loc="best", frameon=False)
    ax_acc.legend(fontsize="small", loc="best", frameon=False)

    plt.tight_layout()

    # Save
    out_len = dir_path / "combined_thinking_length_vs_steering_strength.png"
    out_acc = dir_path / "combined_accuracy_vs_steering_strength.png"

    fig_len.savefig(out_len, dpi=300)
    fig_acc.savefig(out_acc, dpi=300)

    print(f"Saved: {out_len.relative_to(Path.cwd())}")
    print(f"Saved: {out_acc.relative_to(Path.cwd())}")

    plt.close(fig_len)
    plt.close(fig_acc)


# -----------------------------------------------------------------------------
# CLI glue
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:  # noqa: D401 – stylistic preference
    """Parse command‑line arguments."""
    p = argparse.ArgumentParser(
        description=(
            "Aggregate steering‑metric CSVs in a directory and generate "
            "combined summary plots for Thinking Length and Accuracy."
        )
    )
    p.add_argument(
        "dir_path",
        type=Path,
        help="Directory containing the CSV files to aggregate",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        build_combined_plots(args.dir_path.expanduser().resolve())
    except Exception as exc:
        print(f"Error: {exc}")
        raise
