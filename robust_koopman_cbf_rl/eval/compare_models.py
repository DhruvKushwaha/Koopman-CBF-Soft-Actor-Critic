"""Bar-chart comparison of multiple trained agents from eval CSVs."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_METRICS = [
    ("return",         "Episode Return"),
    ("total_cost",     "Episode Cost"),
    ("violation_rate", "Violation Rate"),
    ("min_h_value",    "Min h(z)"),
]


def compare_models(csv_paths, labels, out_path: str):
    n = len(_METRICS)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    x = np.arange(len(labels))
    for ax, (col, title) in zip(axes, _METRICS):
        means, stds = [], []
        for path in csv_paths:
            df = pd.read_csv(path)
            if col in df.columns:
                means.append(float(df[col].mean()))
                stds.append(float(df[col].std()))
            else:
                means.append(float("nan"))
                stds.append(0.0)
        ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare eval results across models.")
    ap.add_argument("--csvs", nargs="+", required=True,
                    help="Paths to eval_results.csv files, one per model")
    ap.add_argument("--labels", nargs="+", required=True,
                    help="Model labels (same order as --csvs)")
    ap.add_argument("--out", required=True, help="Output PNG path")
    a = ap.parse_args()
    if len(a.csvs) != len(a.labels):
        ap.error("--csvs and --labels must have the same number of entries")
    compare_models(a.csvs, a.labels, a.out)
    print(f"Saved comparison → {a.out}")
