"""4-panel filter timing visualisation.

Panels:
  1  Violin plot     — per-call latency distribution per filter (log ms scale)
  2  Scaling plot    — mean latency vs action dimension, one line per filter
  3  CDF             — cumulative latency distribution (real-time feasibility view)
  4  Summary bar     — mean ± std at CartPole dims (u=1) for quick comparison

Usage:
    from robust_koopman_cbf_rl.plots.plot_filter_timing import plot_filter_timing
    plot_filter_timing(timing_df, "filter_timing.png")

    # or via CLI after running benchmark_filters.py:
    python -m robust_koopman_cbf_rl.plots.plot_filter_timing \\
      --csv logs/timing/filter_timing_raw.csv \\
      --out logs/timing/filter_timing.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_FILTER_COLORS = {
    "NullFilter":        "silver",
    "KCBFQPFilter":      "steelblue",
    "PhysicalCBFFilter": "tomato",
}
_FILTER_ORDER = ["NullFilter", "KCBFQPFilter", "PhysicalCBFFilter"]


def _ordered_filters(df: pd.DataFrame) -> list[str]:
    present = df["filter"].unique()
    return [f for f in _FILTER_ORDER if f in present] + \
           [f for f in present if f not in _FILTER_ORDER]


def plot_filter_timing(
    df: pd.DataFrame,
    out_path: str,
    figsize: tuple = (16, 10),
    real_time_ms: float = 10.0,
) -> None:
    """Generate 4-panel filter timing figure.

    Args:
        df:            DataFrame with columns [env, filter, u_dim, trial_ms].
        out_path:      Output PNG path.
        real_time_ms:  Real-time deadline line drawn on violin/CDF panels (ms).
    """
    filters = _ordered_filters(df)
    colors = [_FILTER_COLORS.get(f, "mediumpurple") for f in filters]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax_vio, ax_scale, ax_cdf, ax_bar = axes.flatten()

    # ── Panel 1: Violin (log scale) ───────────────────────────────────────
    data_violin = [df.loc[df["filter"] == f, "trial_ms"].values for f in filters]
    parts = ax_vio.violinplot(
        data_violin, positions=range(len(filters)),
        showmedians=True, showextrema=False,
    )
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col); pc.set_alpha(0.65)
    parts["cmedians"].set_colors("black"); parts["cmedians"].set_linewidth(1.5)

    ax_vio.set_xticks(range(len(filters)))
    ax_vio.set_xticklabels(filters, rotation=15, ha="right", fontsize=9)
    ax_vio.set_yscale("log")
    ax_vio.axhline(real_time_ms, color="red", linestyle="--", linewidth=1.2,
                   label=f"{real_time_ms:.0f} ms real-time budget")
    ax_vio.set_ylabel("Latency (ms, log scale)")
    ax_vio.set_title("Per-Call Latency Distribution\n(all environments pooled)")
    ax_vio.legend(fontsize=8)
    ax_vio.grid(axis="y", which="both", linestyle=":", alpha=0.4)

    # ── Panel 2: Scaling (mean latency vs u_dim) ──────────────────────────
    u_dims = sorted(df["u_dim"].unique())
    for f, col in zip(filters, colors):
        fdf = df[df["filter"] == f]
        means = [fdf.loc[fdf["u_dim"] == u, "trial_ms"].mean() for u in u_dims]
        stds  = [fdf.loc[fdf["u_dim"] == u, "trial_ms"].std()  for u in u_dims]
        means = np.array(means, dtype=float)
        stds  = np.array(stds,  dtype=float)
        valid = ~np.isnan(means)
        if valid.any():
            ax_scale.plot(np.array(u_dims)[valid], means[valid],
                          marker="o", color=col, label=f, linewidth=2)
            ax_scale.fill_between(
                np.array(u_dims)[valid],
                (means - stds)[valid], (means + stds)[valid],
                color=col, alpha=0.15,
            )
    ax_scale.set_xlabel("Action Dimension (u_dim)")
    ax_scale.set_ylabel("Mean Latency (ms)")
    ax_scale.set_title("Latency Scaling with Action Dimension")
    ax_scale.legend(fontsize=8)
    ax_scale.grid(linestyle=":", alpha=0.4)

    # ── Panel 3: CDF ─────────────────────────────────────────────────────
    for f, col in zip(filters, colors):
        times = df.loc[df["filter"] == f, "trial_ms"].values
        if len(times) == 0:
            continue
        sorted_t = np.sort(times)
        cdf = np.arange(1, len(sorted_t) + 1) / len(sorted_t)
        ax_cdf.plot(sorted_t, cdf, color=col, label=f, linewidth=2)
    ax_cdf.axvline(real_time_ms, color="red", linestyle="--", linewidth=1.2,
                   label=f"{real_time_ms:.0f} ms budget")
    ax_cdf.set_xlabel("Latency (ms)")
    ax_cdf.set_ylabel("Cumulative Fraction")
    ax_cdf.set_title("Latency CDF\n(fraction of calls within budget)")
    ax_cdf.set_xscale("log")
    ax_cdf.set_ylim(0, 1.02)
    ax_cdf.legend(fontsize=8)
    ax_cdf.grid(linestyle=":", alpha=0.4)

    # ── Panel 4: Summary bar (smallest u_dim — CartPole-like) ─────────────
    u_min = int(df["u_dim"].min())
    bar_means, bar_stds = [], []
    for f in filters:
        sub = df.loc[(df["filter"] == f) & (df["u_dim"] == u_min), "trial_ms"]
        bar_means.append(sub.mean() if len(sub) else float("nan"))
        bar_stds.append(sub.std()  if len(sub) else 0.0)
    x = np.arange(len(filters))
    ax_bar.bar(x, bar_means, yerr=bar_stds, color=colors, capsize=6,
               error_kw={"elinewidth": 1.5}, edgecolor="gray", linewidth=0.5)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(filters, rotation=15, ha="right", fontsize=9)
    ax_bar.set_ylabel("Mean Latency (ms)")
    ax_bar.set_title(f"Mean Latency at u_dim={u_min}\n(CartPole-scale; note: Physical uses trivial dynamics)")
    ax_bar.axhline(real_time_ms, color="red", linestyle="--", linewidth=1.0,
                   label=f"{real_time_ms:.0f} ms budget")
    ax_bar.legend(fontsize=8)
    ax_bar.grid(axis="y", linestyle=":", alpha=0.4)

    # Annotate bars with values
    for xi, (m, s) in enumerate(zip(bar_means, bar_stds)):
        if not np.isnan(m):
            ax_bar.text(xi, m + s + 0.01, f"{m:.3f}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")

    fig.suptitle("Safety Filter Per-Iteration Latency Comparison\n"
                 "(PhysicalCBF timing uses a trivial numpy dynamics; "
                 "real physics will be slower)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Filter timing plot → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plot filter timing from benchmark CSV.")
    ap.add_argument("--csv", required=True, help="filter_timing_raw.csv from benchmark_filters")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--real_time_ms", type=float, default=10.0)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    plot_filter_timing(df, a.out, real_time_ms=a.real_time_ms)
