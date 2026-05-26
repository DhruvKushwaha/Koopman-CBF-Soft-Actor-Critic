"""4-panel training dashboard: return, cost, CBF gap, intervention rate."""
from __future__ import annotations
import warnings
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PANELS = [
    ("ep_return",         "Episode Return",    "return"),
    ("ep_cost",           "Episode Cost",      "cost"),
    ("cbf_gap_mean",      "Mean CBF Gap",      "gap"),
    ("intervention_rate", "Intervention Rate", "rate"),
]


def plot_training(csv_paths, labels, out_path: str, smooth: int = 10):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    flat_axes = axes.flatten()
    for path, label in zip(csv_paths, labels):
        df = pd.read_csv(path)
        for ax, (col, title, ylabel) in zip(flat_axes, _PANELS):
            if "step" not in df.columns or col not in df.columns:
                warnings.warn(f"plot_training: missing '{col}' or 'step' in {path}; skipping.")
                continue
            y = df[col].rolling(smooth, min_periods=1).mean()
            ax.plot(df["step"], y, label=label)
    for ax, (_, title, ylabel) in zip(flat_axes, _PANELS):
        ax.set_title(title)
        ax.set_xlabel("steps")
        ax.set_ylabel(ylabel)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
