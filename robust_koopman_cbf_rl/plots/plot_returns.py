"""Plot mean +/- std return curves across seeds."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt


def plot_returns(csv_paths, labels, out_path: str, smooth: int = 10):
    fig, ax = plt.subplots(figsize=(6, 4))
    for path, label in zip(csv_paths, labels):
        df = pd.read_csv(path)
        if "ep_return" not in df.columns or "step" not in df.columns:
            import warnings
            warnings.warn(f"plot_returns: missing 'ep_return' or 'step' column in {path}; skipping.")
            continue
        y = df["ep_return"].rolling(smooth, min_periods=1).mean()
        ax.plot(df["step"], y, label=label)
    ax.set_xlabel("steps"); ax.set_ylabel("return"); ax.legend()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
