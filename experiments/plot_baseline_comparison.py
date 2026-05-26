"""Standalone plotting script: scan a log root, load eval CSVs, produce comparison figures.

Usage:
    python -m experiments.plot_baseline_comparison \\
      --log_root logs/cartpole_stab \\
      --out_dir  logs/cartpole_stab/plots
"""
from __future__ import annotations
import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_LABEL_MAP = {
    "sac_baseline": "SAC",
    "ppo_baseline": "PPO",
    "sac_penalty": "SAC+Penalty",
    "sac_lagrangian": "SAC+Lagrangian",
    "lqr": "LQR",
    "pid": "PID",
    "sac_run1": "KCBF-SAC",
    "ppo_run1": "KCBF-PPO",
}

_KCBF_METHODS = {"KCBF-SAC", "KCBF-PPO"}

_METRICS = [
    ("return",            "Episode Return"),
    ("total_cost",        "Total Cost"),
    ("violation_rate",    "Violation Rate"),
    ("episode_violation", "Episode Violation"),
]

_TRAIN_CSV_MAP = {
    "sac_baseline": "sac_baseline.csv",
    "ppo_baseline": "ppo_baseline.csv",
    "sac_penalty": "sac_penalty.csv",
    "sac_lagrangian": "sac_lagrangian.csv",
    "sac_run1": "sac_kcbf.csv",
    "ppo_run1": "ppo_kcbf.csv",
}


def _scan_eval_csvs(log_root: Path):
    """Return list of (label, csv_path) for all found eval_final.csv files."""
    found = []
    for subdir, label in _LABEL_MAP.items():
        p = log_root / subdir / "eval_final.csv"
        if p.exists():
            found.append((label, p))
    return found


def _make_bar_chart(found, out_path: Path):
    """4-panel bar chart: return, total_cost, violation_rate, episode_violation."""
    labels = [lbl for lbl, _ in found]
    colors = ["#4a90d9" if lbl in _KCBF_METHODS else "#aaaaaa" for lbl in labels]

    fig, axes = plt.subplots(1, len(_METRICS), figsize=(4 * len(_METRICS), 4))
    if len(_METRICS) == 1:
        axes = [axes]

    x = np.arange(len(labels))
    for ax, (col, title) in zip(axes, _METRICS):
        means, stds = [], []
        for _, csv_path in found:
            df = pd.read_csv(csv_path)
            if col in df.columns:
                means.append(float(df[col].mean()))
                stds.append(float(df[col].std()))
            else:
                means.append(float("nan"))
                stds.append(0.0)
        # Replace nan std with 0 so error bars don't crash
        stds_clean = [0.0 if np.isnan(s) else s for s in stds]
        ax.bar(x, means, yerr=stds_clean, capsize=4, color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(title)
        ax.set_ylabel(col)

    fig.suptitle("Baseline Comparison", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    print(f"Bar chart saved → {out_path}")


def _make_training_curves(log_root: Path, out_path: Path):
    """Load training CSV logs and call plot_training."""
    from robust_koopman_cbf_rl.plots.plot_training import plot_training
    train_csvs = []
    train_labels = []
    for subdir, csv_name in _TRAIN_CSV_MAP.items():
        p = log_root / subdir / csv_name
        label = _LABEL_MAP.get(subdir, subdir)
        if p.exists():
            train_csvs.append(str(p))
            train_labels.append(label)
    if not train_csvs:
        print("No training CSV logs found; skipping training curves.")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_training(train_csvs, train_labels, str(out_path))
    print(f"Training curves saved → {out_path}")


def _make_summary_csv(found, out_path: Path):
    """Save comparison_summary.csv with mean ± std for key metrics."""
    rows = []
    for label, csv_path in found:
        df = pd.read_csv(csv_path)
        row = {"method": label}
        for col, _ in _METRICS:
            if col in df.columns:
                row[f"{col}_mean"] = float(df[col].mean())
                row[f"{col}_std"] = float(df[col].std())
            else:
                row[f"{col}_mean"] = float("nan")
                row[f"{col}_std"] = float("nan")
        rows.append(row)
    summary = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(str(out_path), index=False)
    print(f"Summary CSV saved → {out_path}")


def main(log_root, out_dir):
    log_root = Path(log_root)
    out_dir = Path(out_dir)

    found = _scan_eval_csvs(log_root)
    if not found:
        print(f"No eval_final.csv files found under {log_root}. Nothing to plot.")
        return

    print(f"Found {len(found)} baselines: {[lbl for lbl, _ in found]}")

    _make_bar_chart(found, out_dir / "comparison_bar.png")
    _make_summary_csv(found, out_dir / "comparison_summary.csv")

    try:
        _make_training_curves(log_root, out_dir / "training_curves.png")
    except Exception as exc:
        warnings.warn(f"Training curves failed: {exc}")

    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plot baseline comparison from eval CSVs.")
    ap.add_argument("--log_root", required=True, help="Root directory containing baseline subdirs")
    ap.add_argument("--out_dir", required=True, help="Directory to save output figures")
    a = ap.parse_args()
    main(a.log_root, a.out_dir)
