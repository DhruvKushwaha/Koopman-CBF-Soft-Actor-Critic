"""Aggregate multi-seed eval results into mean ± std; save plots and summary CSV.

Expects layout produced by run_sweep.py:
    {log_root}/{env_name}/{baseline}/seed_{seed}/eval_final.csv
    {log_root}/{env_name}/lqr/eval_final.csv           (no seed subdirectory)

Outputs:
    {log_root}/aggregate_summary.csv   — one row per (env, baseline)
    {log_root}/aggregate_comparison.png — bar charts with error bars per env

Usage:
    python experiments/aggregate_results.py --log_root logs/sweep
    python experiments/aggregate_results.py --log_root logs/sweep --out_dir logs/figures
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABEL_MAP: dict[str, str] = {
    "sac": "SAC",
    "ppo": "PPO",
    "sac_penalty": "SAC+Penalty",
    "sac_lagrangian": "SAC+Lagrangian",
    "lqr": "LQR",
    "pid": "PID",
    "sac_run1": "KCBF-SAC",
    "ppo_run1": "KCBF-PPO",
}

# Preferred display order (missing entries go at the end)
_ORDER = ["LQR", "PID", "SAC", "PPO", "SAC+Penalty", "SAC+Lagrangian",
          "KCBF-SAC", "KCBF-PPO"]

_KCBF_COLOR = "steelblue"
_BASELINE_COLOR = "lightsteelblue"
_CLASSICAL_COLOR = "silver"


def _load_eval(csv_path: Path) -> tuple[float, float]:
    try:
        df = pd.read_csv(csv_path)
        ret = float(df["return"].mean()) if "return" in df.columns else float("nan")
        cost = float(df["total_cost"].mean()) if "total_cost" in df.columns else float("nan")
        return ret, cost
    except Exception:
        return float("nan"), float("nan")


def _bar_color(label: str) -> str:
    if "KCBF" in label:
        return _KCBF_COLOR
    if label in ("LQR", "PID"):
        return _CLASSICAL_COLOR
    return _BASELINE_COLOR


def aggregate(log_root: str, out_dir: str | None = None) -> pd.DataFrame:
    root = Path(log_root)
    out_root = Path(out_dir) if out_dir else root
    out_root.mkdir(parents=True, exist_ok=True)

    # Collect: {env_name: {baseline: {seed_or_0: (return, cost)}}}
    data: dict[str, dict[str, dict[int, tuple[float, float]]]] = {}

    for env_dir in sorted(root.iterdir()):
        if not env_dir.is_dir() or env_dir.name.startswith("."):
            continue
        env_name = env_dir.name

        for bl_dir in sorted(env_dir.iterdir()):
            if not bl_dir.is_dir():
                continue
            bl_name = bl_dir.name

            seed_dirs = sorted(
                d for d in bl_dir.iterdir()
                if d.is_dir() and d.name.startswith("seed_")
            )
            if seed_dirs:
                seeds_data = {}
                for sd in seed_dirs:
                    try:
                        seed = int(sd.name.split("_")[1])
                    except (IndexError, ValueError):
                        continue
                    seeds_data[seed] = _load_eval(sd / "eval_final.csv")
            else:
                f = bl_dir / "eval_final.csv"
                if not f.exists():
                    continue
                seeds_data = {0: _load_eval(f)}

            if not seeds_data:
                continue
            data.setdefault(env_name, {})[bl_name] = seeds_data

    if not data:
        print(f"No eval_final.csv files found under {root}")
        return pd.DataFrame()

    # Build summary dataframe
    rows = []
    for env_name, baselines in data.items():
        for bl_name, seeds_data in baselines.items():
            rets = [v[0] for v in seeds_data.values() if not np.isnan(v[0])]
            costs = [v[1] for v in seeds_data.values() if not np.isnan(v[1])]
            rows.append({
                "env": env_name,
                "baseline": bl_name,
                "label": LABEL_MAP.get(bl_name, bl_name),
                "n_seeds": len(seeds_data),
                "mean_return": np.mean(rets) if rets else float("nan"),
                "std_return": np.std(rets) if len(rets) > 1 else 0.0,
                "mean_cost": np.mean(costs) if costs else float("nan"),
                "std_cost": np.std(costs) if len(costs) > 1 else 0.0,
            })

    summary = pd.DataFrame(rows)
    csv_path = out_root / "aggregate_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"Summary → {csv_path}")

    # Plot
    env_names = sorted(summary["env"].unique())
    n_envs = len(env_names)
    fig, axes = plt.subplots(n_envs, 2, figsize=(13, 4 * n_envs), squeeze=False)
    fig.suptitle("Baseline Comparison (mean ± std over seeds)", fontsize=13, y=1.01)

    for i, env_name in enumerate(env_names):
        edf = summary[summary["env"] == env_name].copy()
        # Sort by preferred display order
        edf["_ord"] = edf["label"].map(
            lambda lbl: _ORDER.index(lbl) if lbl in _ORDER else len(_ORDER)
        )
        edf = edf.sort_values("_ord").reset_index(drop=True)
        labels = edf["label"].tolist()
        x = np.arange(len(labels))
        colors = [_bar_color(lbl) for lbl in labels]

        for j, (val_col, err_col, ylabel) in enumerate([
            ("mean_return", "std_return", "Episode Return"),
            ("mean_cost", "std_cost", "Episode Cost"),
        ]):
            ax = axes[i, j]
            vals = edf[val_col].values
            errs = edf[err_col].values
            ax.bar(x, vals, yerr=errs, capsize=5, color=colors,
                   error_kw={"elinewidth": 1.5, "ecolor": "dimgray"},
                   edgecolor="gray", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(f"{env_name}", fontsize=11, fontweight="bold")
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
            ax.grid(axis="y", linestyle=":", alpha=0.4)

    plt.tight_layout()
    plot_path = out_root / "aggregate_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot    → {plot_path}")

    _print_table(summary)
    return summary


def _print_table(df: pd.DataFrame) -> None:
    print("\n=== Aggregate Summary ===")
    for env_name in sorted(df["env"].unique()):
        print(f"\n  {env_name}")
        edf = df[df["env"] == env_name]
        for _, row in edf.iterrows():
            seeds_str = f"(n={int(row['n_seeds'])})"
            print(f"    {row['label']:20s}  return={row['mean_return']:+8.2f}±{row['std_return']:5.2f}"
                  f"  cost={row['mean_cost']:7.2f}±{row['std_cost']:5.2f}  {seeds_str}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Aggregate multi-seed sweep results.")
    ap.add_argument("--log_root", required=True, help="Root directory from run_sweep.py")
    ap.add_argument("--out_dir", default=None,
                    help="Output directory for plots/CSV (default: same as log_root)")
    a = ap.parse_args()
    aggregate(a.log_root, a.out_dir)
