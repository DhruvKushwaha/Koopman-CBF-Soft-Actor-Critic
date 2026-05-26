"""Plot CBF intervention rate over training."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_intervention_rate(csv_path, out_path: str):
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    if "intervention_rate" in df.columns:
        ax.plot(df["step"], df["intervention_rate"])
    ax.set_xlabel("steps"); ax.set_ylabel("intervention rate")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
