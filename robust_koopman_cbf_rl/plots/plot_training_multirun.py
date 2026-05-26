"""Multi-seed training curves with mean ± std shading.

Resamples each per-seed training CSV onto a common step grid via linear
interpolation, then plots mean ± std bands across seeds for each method.

4-panel layout:
  1  Episode return         (higher is better)
  2  Episode cost           (lower is better)
  3  Mean CBF gap           (KCBF only; NaN elsewhere → flat 0 with annotation)
  4  Intervention rate      (KCBF only; NaN elsewhere)

Usage:
    from robust_koopman_cbf_rl.plots.plot_training_multirun import plot_training_multirun
    plot_training_multirun(
        csv_groups={
            "KCBF-SAC":    ["seed0/sac_kcbf.csv", "seed1/sac_kcbf.csv"],
            "SAC":         ["seed0/sac_baseline.csv", "seed1/sac_baseline.csv"],
        },
        out_path="training_multirun.png",
    )

CLI:
    python -m robust_koopman_cbf_rl.plots.plot_training_multirun \\
      --log_root logs/sweep/cartpole_stab \\
      --out training_cartpole_stab.png
"""
from __future__ import annotations
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PANELS = [
    ("ep_return",         "Episode Return",     True),
    ("ep_cost",           "Episode Cost",       True),
    ("cbf_gap_mean",      "Mean CBF Gap",       False),
    ("intervention_rate", "Intervention Rate",  False),
]

_CMAP = plt.get_cmap("tab10")

# Maps baseline folder name → training CSV filename
_TRAIN_CSV_MAP = {
    "sac":            "sac_baseline.csv",
    "ppo":            "ppo_baseline.csv",
    "sac_penalty":    "sac_penalty.csv",
    "sac_lagrangian": "sac_lagrangian.csv",
    "sac_run1":       "sac_kcbf.csv",
    "ppo_run1":       "ppo_kcbf.csv",
    "kcbf_sac":       "sac_kcbf.csv",
    "kcbf_ppo":       "ppo_kcbf.csv",
}

_LABEL_MAP = {
    "sac":            "SAC",
    "ppo":            "PPO",
    "sac_penalty":    "SAC+Penalty",
    "sac_lagrangian": "SAC+Lagrangian",
    "sac_run1":       "KCBF-SAC",
    "ppo_run1":       "KCBF-PPO",
    "kcbf_sac":       "KCBF-SAC",
    "kcbf_ppo":       "KCBF-PPO",
}


def _load_col(csv_path: str, col: str, step_grid: np.ndarray) -> np.ndarray:
    """Read col from CSV and interpolate onto step_grid. Returns NaN array on failure."""
    try:
        df = pd.read_csv(csv_path)
        if col not in df.columns or "step" not in df.columns:
            return np.full(len(step_grid), float("nan"))
        tmp = df[["step", col]].dropna().sort_values("step")
        if len(tmp) < 2:
            return np.full(len(step_grid), float("nan"))
        return np.interp(step_grid, tmp["step"].values, tmp[col].values,
                         left=float("nan"), right=float("nan"))
    except Exception as e:
        warnings.warn(f"plot_training_multirun: could not load {col} from {csv_path}: {e}")
        return np.full(len(step_grid), float("nan"))


def plot_training_multirun(
    csv_groups: dict[str, list[str]],
    out_path: str,
    smooth: int = 20,
    n_points: int = 500,
    alpha_band: float = 0.2,
    figsize: tuple = (14, 8),
) -> None:
    """Plot mean ± std training curves for each group in csv_groups.

    Args:
        csv_groups: {label: [csv_seed0, csv_seed1, ...]}
        out_path:   Output PNG path.
        smooth:     Window for rolling mean applied to each seed's curve before
                    computing mean/std (smooths within-seed noise).
        n_points:   Number of points on shared step grid.
        alpha_band: Opacity of the std shading.
    """
    # Determine global step range from all CSVs
    step_min, step_max = float("inf"), -float("inf")
    for paths in csv_groups.values():
        for p in paths:
            try:
                df = pd.read_csv(p)
                if "step" in df.columns and len(df) > 0:
                    step_min = min(step_min, float(df["step"].min()))
                    step_max = max(step_max, float(df["step"].max()))
            except Exception:
                pass
    if step_min == float("inf"):
        warnings.warn("plot_training_multirun: no valid CSVs found.")
        return
    step_grid = np.linspace(step_min, step_max, n_points)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    flat_axes = axes.flatten()

    for gi, (label, paths) in enumerate(csv_groups.items()):
        color = _CMAP(gi % 10)
        for ai, (col, title, always_show) in enumerate(_PANELS):
            ax = flat_axes[ai]
            curves = []
            for p in paths:
                raw = _load_col(p, col, step_grid)
                if not np.all(np.isnan(raw)):
                    # Smooth within seed
                    valid = ~np.isnan(raw)
                    smoothed = raw.copy()
                    if smooth > 1 and valid.sum() > smooth:
                        kernel = np.ones(smooth) / smooth
                        smoothed[valid] = np.convolve(raw[valid], kernel, mode="same")
                    curves.append(smoothed)
            if not curves:
                continue
            mat = np.stack(curves)  # (n_seeds, n_points)
            mu = np.nanmean(mat, axis=0)
            sigma = np.nanstd(mat, axis=0)
            if np.all(np.isnan(mu)):
                if not always_show:
                    continue
            ax.plot(step_grid, mu, color=color, label=label, linewidth=1.8)
            ax.fill_between(step_grid, mu - sigma, mu + sigma,
                            color=color, alpha=alpha_band)

    for ai, (col, title, _) in enumerate(_PANELS):
        ax = flat_axes[ai]
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Steps", fontsize=9)
        ax.set_ylabel(title, fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        handles, lbls = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)

    fig.suptitle("Training Curves (mean ± std across seeds)", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training multirun plot → {out_path}")


def _discover_csv_groups(log_root: str) -> dict[str, list[str]]:
    """Scan a sweep log directory and build csv_groups from seed subdirectories."""
    root = Path(log_root)
    groups: dict[str, list[str]] = {}
    for bl_dir in sorted(root.iterdir()):
        if not bl_dir.is_dir():
            continue
        bl_name = bl_dir.name
        csv_name = _TRAIN_CSV_MAP.get(bl_name)
        if csv_name is None:
            continue
        label = _LABEL_MAP.get(bl_name, bl_name)
        seed_dirs = sorted(
            d for d in bl_dir.iterdir()
            if d.is_dir() and d.name.startswith("seed_")
        )
        found = [str(sd / csv_name) for sd in seed_dirs if (sd / csv_name).exists()]
        if found:
            groups[label] = found
    return groups


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plot multi-seed training curves.")
    ap.add_argument("--log_root", required=True,
                    help="Sweep sub-directory for ONE environment "
                         "(e.g. logs/sweep/cartpole_stab)")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--smooth", type=int, default=20)
    ap.add_argument("--n_points", type=int, default=500)
    a = ap.parse_args()
    groups = _discover_csv_groups(a.log_root)
    if not groups:
        print("No training CSVs found.")
    else:
        plot_training_multirun(groups, a.out, smooth=a.smooth, n_points=a.n_points)
