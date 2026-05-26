"""Plot violation rate vs steps."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_violations(csv_paths, labels, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    for path, label in zip(csv_paths, labels):
        df = pd.read_csv(path)
        if "ep_cost" in df.columns:
            ax.plot(df["step"], df["ep_cost"], label=label)
    ax.set_xlabel("steps"); ax.set_ylabel("episode cost"); ax.legend()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
