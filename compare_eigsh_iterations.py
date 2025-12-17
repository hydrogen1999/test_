#!/usr/bin/env python3
"""Aggregate eigsh iteration counts across SpecAnn runs.

The script scans the `results/SpecAnn` and `results/SpecAnn-cold` folders,
reads every CSV inside their `alphas/` subfolders, and computes:

1. The mean `eigsh_iterations` per graph (averaging across alphas).
2. The mean of those values for each graph size.

It prints the resulting table, writes it to
`results/eigsh_iterations_summary.csv`, and saves a comparison plot at
`results/eigsh_iterations_vs_size.png`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
MODELS = ("SpecAnn", "SpecAnn-cold")
SUMMARY_CSV = RESULTS_DIR / "eigsh_iterations_summary.csv"
PLOT_FILE = RESULTS_DIR / "eigsh_iterations_vs_size.png"


def collect_per_graph_averages() -> pd.DataFrame:
    """Return per-graph average eigsh iterations for each model."""
    records: list[dict[str, object]] = []
    for model in MODELS:
        alpha_dir = RESULTS_DIR / model / "alphas"
        if not alpha_dir.is_dir():
            raise FileNotFoundError(f"Missing directory: {alpha_dir}")
        csv_files = sorted(alpha_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {alpha_dir}")

        for csv_path in csv_files:
            try:
                size = int(csv_path.stem.split("_")[1])
            except (IndexError, ValueError) as exc:
                raise ValueError(
                    f"Could not parse size from filename '{csv_path.name}'"
                ) from exc

            df = pd.read_csv(csv_path)
            if "eigsh_iterations" not in df.columns:
                raise ValueError(
                    f"'eigsh_iterations' column missing in {csv_path}"
                )

            avg_iters = df["eigsh_iterations"].mean()
            records.append(
                {
                    "model": model,
                    "size": size,
                    "graph_file": csv_path.name,
                    "avg_iterations": avg_iters,
                }
            )

    return pd.DataFrame.from_records(records)


def aggregate_by_size(per_graph_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate averages per size and return pivoted table."""
    grouped = (
        per_graph_df.groupby(["model", "size"])["avg_iterations"]
        .mean()
        .reset_index()
    )
    pivot = (
        grouped.pivot(index="size", columns="model", values="avg_iterations")
        .sort_index()
    )
    pivot.index.name = "size"
    return pivot


def plot_summary(summary_df: pd.DataFrame) -> None:
    """Generate and save the comparison plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in summary_df.columns:
        ax.plot(
            summary_df.index,
            summary_df[model],
            marker="o",
            label=model,
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Graph size (nodes)")
    ax.set_ylabel("Average eigsh iterations")
    ax.set_title("Average eigsh iterations vs graph size")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=200)
    plt.close(fig)


def main() -> int:
    per_graph_df = collect_per_graph_averages()
    summary_df = aggregate_by_size(per_graph_df)

    summary_df.to_csv(SUMMARY_CSV)
    plot_summary(summary_df)

    print("Average eigsh iterations per graph size:")
    print(summary_df)
    print(f"\nSummary saved to: {SUMMARY_CSV}")
    print(f"Plot saved to: {PLOT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

