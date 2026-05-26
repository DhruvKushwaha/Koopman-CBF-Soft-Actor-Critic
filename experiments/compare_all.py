"""Unified comparison of all baselines + KCBF variants across all environments.

Reads eval_final.csv files from both a baseline sweep directory and a KCBF sweep
directory, aggregates over seeds, and generates:

  {out_dir}/
    all_methods_comparison.png       — return + cost + violation bar charts per env
    pareto_{env}.png                 — safety-efficiency Pareto scatter per env
    kcbf_diagnostics.png             — KCBF intervention rate and h_min per env
    training_curves_{env}.png        — multi-seed mean±std training curves per env
    aggregate_all_summary.csv        — one row per (env, method)
    latex_table.txt                  — LaTeX tabular for the paper

Usage:
    # After running run_sweep.py and run_kcbf_sweep.py:
    python experiments/compare_all.py \\
      --baseline_root logs/sweep \\
      --kcbf_root     logs/kcbf_sweep \\
      --out_dir       logs/comparison

    # Only baselines (no KCBF yet):
    python experiments/compare_all.py \\
      --baseline_root logs/sweep \\
      --out_dir       logs/comparison
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

_ROOT = Path(__file__).parent.parent

# Display order and colors
_METHOD_ORDER = [
    "LQR", "PID",
    "SAC", "PPO",
    "SAC+Penalty", "SAC+Lagrangian",
    "KCBF-SAC", "KCBF-PPO",
]

_COLORS = {
    "LQR":            "#b0b0b0",
    "PID":            "#909090",
    "SAC":            "#aec7e8",
    "PPO":            "#98df8a",
    "SAC+Penalty":    "#ffbb78",
    "SAC+Lagrangian": "#ff9896",
    "KCBF-SAC":       "#1f77b4",
    "KCBF-PPO":       "#2ca02c",
}

_KCBF_METHODS = {"KCBF-SAC", "KCBF-PPO"}

# Baseline folder-name → display label
_LABEL_MAP = {
    "sac":            "SAC",
    "ppo":            "PPO",
    "sac_penalty":    "SAC+Penalty",
    "sac_lagrangian": "SAC+Lagrangian",
    "lqr":            "LQR",
    "pid":            "PID",
    "kcbf_sac":       "KCBF-SAC",
    "kcbf_ppo":       "KCBF-PPO",
    "sac_run1":       "KCBF-SAC",
    "ppo_run1":       "KCBF-PPO",
}

# Baseline folder-name → training CSV filename (for multi-run curves)
_TRAIN_CSV = {
    "sac":            "sac_baseline.csv",
    "ppo":            "ppo_baseline.csv",
    "sac_penalty":    "sac_penalty.csv",
    "sac_lagrangian": "sac_lagrangian.csv",
    "kcbf_sac":       "sac_kcbf.csv",
    "kcbf_ppo":       "ppo_kcbf.csv",
}

# Columns read from eval_final.csv
_EVAL_METRICS = [
    "return", "total_cost", "violation_rate",
    "episode_violation", "min_h_value", "intervention_rate",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_eval_csv(csv_path: Path) -> dict[str, float]:
    """Return per-metric means from one eval_final.csv (averaged over episodes)."""
    try:
        df = pd.read_csv(csv_path)
        result = {}
        for col in _EVAL_METRICS:
            if col not in df.columns:
                continue
            vals = df[col].dropna().values
            if len(vals) > 0:
                result[col] = float(np.mean(vals))
        return result
    except Exception:
        return {}


def _scan_root(root: Path) -> dict[str, dict[str, list[Path]]]:
    """Scan a sweep directory → {env_name: {label: [seed_eval_csvs]}}."""
    data: dict[str, dict[str, list[Path]]] = {}
    if not root.exists():
        return data
    for env_dir in sorted(root.iterdir()):
        if not env_dir.is_dir():
            continue
        env_name = env_dir.name
        for bl_dir in sorted(env_dir.iterdir()):
            if not bl_dir.is_dir():
                continue
            bl_name = bl_dir.name
            label = _LABEL_MAP.get(bl_name)
            if label is None:
                continue
            seed_dirs = [d for d in bl_dir.iterdir()
                         if d.is_dir() and d.name.startswith("seed_")]
            if seed_dirs:
                csvs = [sd / "eval_final.csv" for sd in seed_dirs
                        if (sd / "eval_final.csv").exists()]
            else:
                f = bl_dir / "eval_final.csv"
                csvs = [f] if f.exists() else []
            if csvs:
                data.setdefault(env_name, {}).setdefault(label, []).extend(csvs)
    return data


def _build_summary(data_baseline: dict, data_kcbf: dict) -> pd.DataFrame:
    """Merge both sweep directories into a summary DataFrame."""
    merged: dict[str, dict[str, list[Path]]] = {}
    for env_name, bls in data_baseline.items():
        merged.setdefault(env_name, {}).update(bls)
    for env_name, bls in data_kcbf.items():
        for label, paths in bls.items():
            merged.setdefault(env_name, {}).setdefault(label, []).extend(paths)

    rows = []
    for env_name, bls in merged.items():
        for label, csv_paths in bls.items():
            # Collect per-seed metric dicts
            seed_metrics: dict[str, list[float]] = {}
            for p in csv_paths:
                m = _load_eval_csv(p)
                for k, v in m.items():
                    if not np.isnan(v):
                        seed_metrics.setdefault(k, []).append(v)

            row: dict = {"env": env_name, "method": label, "n_seeds": len(csv_paths)}
            for col in _EVAL_METRICS:
                vals = seed_metrics.get(col, [])
                row[f"mean_{col}"] = np.mean(vals) if vals else float("nan")
                row[f"std_{col}"] = np.std(vals) if len(vals) > 1 else 0.0
            # Stable aliases used by plots and LaTeX table
            row["mean_cost"] = row["mean_total_cost"]
            row["std_cost"]  = row["std_total_cost"]
            rows.append(row)
    return pd.DataFrame(rows)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sorted_methods(edf: pd.DataFrame) -> pd.DataFrame:
    edf = edf.copy()
    edf["_ord"] = edf["method"].map(
        lambda m: _METHOD_ORDER.index(m) if m in _METHOD_ORDER else len(_METHOD_ORDER)
    )
    return edf.sort_values("_ord").reset_index(drop=True)


def _annotate_bars(ax, x, vals, errs, fmt=".1f"):
    for xi, (v, e) in enumerate(zip(vals, errs)):
        if not np.isnan(v):
            ax.text(xi, v + e + abs(v) * 0.01 + 0.01,
                    format(v, fmt), ha="center", va="bottom", fontsize=7)


# ── Plot 1: Return / Cost / Violation bar chart ───────────────────────────────

def _comparison_bar(summary: pd.DataFrame, out_path: Path) -> None:
    env_names = sorted(summary["env"].unique())
    n_env = len(env_names)
    fig, axes = plt.subplots(n_env, 3, figsize=(18, 4.5 * n_env), squeeze=False)
    fig.suptitle(
        "All Methods — Return, Safety Cost, and Violation Rate\n(mean ± std across seeds)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    panels = [
        ("mean_return",          "std_return",          "Episode Return ↑",       ".1f"),
        ("mean_total_cost",      "std_total_cost",      "Episode Cost ↓",          ".1f"),
        ("mean_violation_rate",  "std_violation_rate",  "Violation Rate ↓",        ".3f"),
    ]

    for i, env_name in enumerate(env_names):
        edf = _sorted_methods(summary[summary["env"] == env_name])
        methods = edf["method"].tolist()
        x = np.arange(len(methods))
        cols = [_COLORS.get(m, "mediumpurple") for m in methods]

        for j, (val_col, err_col, title, fmt) in enumerate(panels):
            ax = axes[i, j]
            if val_col not in edf.columns:
                ax.set_visible(False)
                continue
            vals = edf[val_col].values
            errs = edf[err_col].values if err_col in edf.columns else np.zeros_like(vals)
            ax.bar(x, vals, yerr=errs, color=cols, capsize=5,
                   error_kw={"elinewidth": 1.5}, edgecolor="gray", linewidth=0.4)
            _annotate_bars(ax, x, vals, errs, fmt=fmt)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=9)
            ax.set_ylabel(title, fontsize=10)
            ax.set_title(f"{env_name} — {title}", fontsize=11, fontweight="bold")
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
            ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison chart  → {out_path}")


# ── Plot 2: Safety-Efficiency Pareto scatter ──────────────────────────────────

def _pareto_scatter(summary: pd.DataFrame, out_dir: Path) -> None:
    """Per-env scatter: return (x) vs violation_rate (y). KCBF highlighted."""
    env_names = sorted(summary["env"].unique())
    if "mean_violation_rate" not in summary.columns:
        return

    n_env = len(env_names)
    fig, axes = plt.subplots(1, n_env, figsize=(6 * n_env, 5), squeeze=False)
    fig.suptitle(
        "Safety-Efficiency Pareto: Return vs. Violation Rate\n"
        "(upper-left = high reward, low violation = best)",
        fontsize=13, fontweight="bold",
    )

    for i, env_name in enumerate(env_names):
        ax = axes[0, i]
        edf = summary[summary["env"] == env_name].dropna(
            subset=["mean_return", "mean_violation_rate"]
        )
        for _, row in edf.iterrows():
            m = row["method"]
            color = _COLORS.get(m, "mediumpurple")
            is_kcbf = m in _KCBF_METHODS
            marker = "*" if is_kcbf else "o"
            size = 220 if is_kcbf else 100
            zorder = 4 if is_kcbf else 3
            ax.scatter(row["mean_return"], row["mean_violation_rate"],
                       color=color, marker=marker, s=size, zorder=zorder,
                       edgecolors="black", linewidths=0.8 if is_kcbf else 0.4,
                       label=m)
            # Error bars
            xerr = row.get("std_return", 0.0) or 0.0
            yerr = row.get("std_violation_rate", 0.0) or 0.0
            ax.errorbar(row["mean_return"], row["mean_violation_rate"],
                        xerr=xerr, yerr=yerr,
                        fmt="none", color=color, alpha=0.5, linewidth=1.0, capsize=3)
            ax.annotate(m, (row["mean_return"], row["mean_violation_rate"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=7)

        ax.set_xlabel("Episode Return ↑", fontsize=11)
        ax.set_ylabel("Violation Rate ↓", fontsize=11)
        ax.set_title(env_name, fontsize=12, fontweight="bold")
        ax.grid(linestyle=":", alpha=0.4)
        # Shade the ideal region (upper-left quadrant from origin)
        ax.axhline(0, color="green", linestyle="--", linewidth=0.8, alpha=0.4)

    fig.tight_layout()
    out_path = out_dir / "pareto_safety_efficiency.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Pareto scatter    → {out_path}")


# ── Plot 3: KCBF CBF diagnostics ─────────────────────────────────────────────

def _kcbf_diagnostics(summary: pd.DataFrame, out_dir: Path) -> None:
    """Bar charts of KCBF-specific metrics: intervention_rate and h_min."""
    kcbf_df = summary[summary["method"].isin(_KCBF_METHODS)].copy()
    if kcbf_df.empty:
        return

    diag_panels = [
        ("mean_intervention_rate", "std_intervention_rate",
         "CBF Intervention Rate",
         "Fraction of steps where QP filter overrides policy action"),
        ("mean_min_h_value", "std_min_h_value",
         "Min Barrier Value h(x)",
         "h < 0 → safety violation entered; h ≥ 0 → always safe"),
    ]

    available = [(vc, ec, t, d) for vc, ec, t, d in diag_panels if vc in kcbf_df.columns]
    if not available:
        return

    env_names = sorted(kcbf_df["env"].unique())
    n_env = len(env_names)
    n_panels = len(available)

    fig, axes = plt.subplots(n_panels, n_env,
                             figsize=(max(4, 3.5 * n_env), 4.5 * n_panels),
                             squeeze=False)
    fig.suptitle("KCBF Safety Filter Diagnostics", fontsize=13, fontweight="bold")

    for j, (val_col, err_col, title, desc) in enumerate(available):
        is_hmin = "h_value" in val_col

        for i, env_name in enumerate(env_names):
            ax = axes[j, i]
            sub = kcbf_df[kcbf_df["env"] == env_name]
            if sub.empty:
                ax.set_visible(False)
                continue

            methods = sub["method"].tolist()
            x = np.arange(len(methods))
            vals = sub[val_col].values.astype(float)
            errs = (sub[err_col].values.astype(float)
                    if err_col in sub.columns else np.zeros_like(vals))

            # Color: h_min uses safety semantics (green/red), others use method palette
            if is_hmin:
                bar_cols = ["#d62728" if v < 0 else "#2ca02c" for v in vals]
            else:
                bar_cols = [_COLORS.get(m, "steelblue") for m in methods]

            bars = ax.bar(x, vals, yerr=errs, color=bar_cols, capsize=5,
                          error_kw={"elinewidth": 1.5}, edgecolor="gray",
                          linewidth=0.5, zorder=3)

            # bar_label places text at bar tip regardless of sign
            ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=8)

            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
            ax.set_title(env_name, fontsize=11, fontweight="bold")
            ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)

            # y-limits: add 20% headroom above max tip, 20% below min dip
            max_tip = float(np.nanmax(vals + errs)) if len(vals) else 0.0
            min_dip = float(np.nanmin(vals - errs)) if len(vals) else 0.0
            span = max(abs(max_tip - min_dip), 1e-6)

            if is_hmin:
                # Extra headroom for labels; always show y=0
                ax.set_ylim(min(min_dip - 0.25 * span, -0.5),
                            max_tip + 0.35 * span)
                ax.axhline(0, color="black", linestyle="--", linewidth=1.0,
                           alpha=0.7, label="Safety boundary (h=0)", zorder=2)
                # Shade safe region
                ax.axhspan(0, ax.get_ylim()[1], alpha=0.06, color="green", zorder=1)
                if i == n_env - 1:
                    ax.legend(fontsize=7, loc="upper right")
            else:
                # Intervention rate: always [0, 1] domain, pad top for labels
                ax.set_ylim(0, max(1.05, max_tip + 0.25 * span))

            if i == 0:
                ax.set_ylabel(f"{title}\n({desc})", fontsize=9)
            else:
                ax.set_ylabel(title, fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = out_dir / "kcbf_diagnostics.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"KCBF diagnostics  → {out_path}")


# ── Plot 4: Episode violation fraction (at least one violation per episode) ───

def _episode_violation_bar(summary: pd.DataFrame, out_dir: Path) -> None:
    """Fraction of episodes with ≥1 constraint violation — tighter safety metric."""
    if "mean_episode_violation" not in summary.columns:
        return

    env_names = sorted(summary["env"].unique())
    n_env = len(env_names)
    fig, axes = plt.subplots(1, n_env, figsize=(5 * n_env, 4.5), squeeze=False)
    fig.suptitle(
        "Episode Violation Fraction\n(fraction of eval episodes with ≥1 constraint violation)",
        fontsize=13, fontweight="bold",
    )

    for i, env_name in enumerate(env_names):
        ax = axes[0, i]
        edf = _sorted_methods(summary[summary["env"] == env_name])
        methods = edf["method"].tolist()
        x = np.arange(len(methods))
        cols = [_COLORS.get(m, "mediumpurple") for m in methods]
        vals = edf["mean_episode_violation"].values
        errs = edf.get("std_episode_violation", pd.Series(np.zeros(len(vals)))).values
        ax.bar(x, vals, yerr=errs, color=cols, capsize=5,
               error_kw={"elinewidth": 1.5}, edgecolor="gray", linewidth=0.4)
        _annotate_bars(ax, x, vals, errs, fmt=".2f")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Episode Violation Fraction ↓", fontsize=10)
        ax.set_title(env_name, fontsize=11, fontweight="bold")
        ax.axhline(1.0, color="red", linestyle="--", linewidth=0.8, alpha=0.4,
                   label="All episodes violated")
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    out_path = out_dir / "episode_violation_fraction.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Episode violation → {out_path}")


# ── Training curves ───────────────────────────────────────────────────────────

def _training_curves(baseline_root: Path | None, kcbf_root: Path | None,
                     env_name: str, out_path: Path) -> None:
    """Multi-seed training curves for one environment."""
    try:
        from robust_koopman_cbf_rl.plots.plot_training_multirun import (
            plot_training_multirun, _TRAIN_CSV_MAP, _LABEL_MAP as _LM,
        )
    except ImportError as e:
        warnings.warn(f"plot_training_multirun import failed: {e}")
        return

    csv_groups: dict[str, list[str]] = {}

    def _add_from_root(root: Path) -> None:
        env_dir = root / env_name
        if not env_dir.exists():
            return
        for bl_dir in sorted(env_dir.iterdir()):
            if not bl_dir.is_dir():
                continue
            bl_name = bl_dir.name
            csv_name = _TRAIN_CSV.get(bl_name) or _TRAIN_CSV_MAP.get(bl_name)
            if not csv_name:
                continue
            label = _LABEL_MAP.get(bl_name, bl_name)
            seed_dirs = sorted(
                d for d in bl_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")
            )
            found = [str(sd / csv_name) for sd in seed_dirs if (sd / csv_name).exists()]
            if found:
                csv_groups.setdefault(label, []).extend(found)

    if baseline_root:
        _add_from_root(baseline_root)
    if kcbf_root:
        _add_from_root(kcbf_root)

    if csv_groups:
        plot_training_multirun(csv_groups, str(out_path))
    else:
        warnings.warn(f"No training CSVs found for {env_name}")


# ── LaTeX table ───────────────────────────────────────────────────────────────

def _latex_table(summary: pd.DataFrame, out_path: Path) -> None:
    has_viol = "mean_violation_rate" in summary.columns
    col_spec = "llrrrr" if not has_viol else "llrrrrr"
    header_cols = r"Return $\uparrow$ & Cost $\downarrow$ & Seeds \\"
    if has_viol:
        header_cols = r"Return $\uparrow$ & Cost $\downarrow$ & Viol. Rate $\downarrow$ & Seeds \\"

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Comparison of baselines and KCBF variants (mean $\pm$ std over 3 seeds)}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"Environment & Method & {header_cols}",
        r"\midrule",
    ]
    current_env = None
    for _, row in summary.sort_values(["env", "method"]).iterrows():
        env = row["env"].replace("_", r"\_")
        if env != current_env:
            if current_env is not None:
                lines.append(r"\midrule")
            current_env = env
            env_str = r"\textbf{" + env + "}"
        else:
            env_str = ""
        method = row["method"].replace("+", r"\texttt{+}")
        ret = f"{row['mean_return']:.1f} $\\pm$ {row['std_return']:.1f}"
        cost = f"{row['mean_total_cost']:.2f} $\\pm$ {row['std_total_cost']:.2f}"
        if has_viol and not np.isnan(row.get("mean_violation_rate", float("nan"))):
            viol = f"{row['mean_violation_rate']:.3f} $\\pm$ {row.get('std_violation_rate', 0.0):.3f}"
            lines.append(f"{env_str} & {method} & {ret} & {cost} & {viol} & {int(row['n_seeds'])} \\\\")
        else:
            lines.append(f"{env_str} & {method} & {ret} & {cost} & {int(row['n_seeds'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"LaTeX table       → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Unified comparison of all methods across all environments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--baseline_root", default=None,
                    help="Sweep log root from run_sweep.py")
    ap.add_argument("--kcbf_root", default=None,
                    help="Sweep log root from run_kcbf_sweep.py")
    ap.add_argument("--out_dir", required=True, help="Output directory for all figures")
    ap.add_argument("--skip_training_curves", action="store_true")
    a = ap.parse_args()

    if not a.baseline_root and not a.kcbf_root:
        ap.error("At least one of --baseline_root or --kcbf_root must be specified.")

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_root = Path(a.baseline_root) if a.baseline_root else None
    kcbf_root = Path(a.kcbf_root) if a.kcbf_root else None

    # ── Build summary ──────────────────────────────────────────────────────
    data_bl = _scan_root(baseline_root) if baseline_root else {}
    data_kc = _scan_root(kcbf_root) if kcbf_root else {}
    summary = _build_summary(data_bl, data_kc)

    if summary.empty:
        print("No eval_final.csv files found. Check --baseline_root and --kcbf_root.")
        return

    csv_path = out_dir / "aggregate_all_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"Summary CSV       → {csv_path}")

    # Print table to console
    print("\n=== Results ===")
    for env_name in sorted(summary["env"].unique()):
        print(f"\n  {env_name}")
        sub = summary[summary["env"] == env_name]
        for _, row in sub.iterrows():
            viol = row.get("mean_violation_rate", float("nan"))
            viol_str = f"  viol={viol:.3f}" if not np.isnan(viol) else ""
            print(f"    {row['method']:20s}  "
                  f"return={row['mean_return']:+8.2f}±{row['std_return']:.2f}  "
                  f"cost={row['mean_total_cost']:7.2f}±{row['std_total_cost']:.2f}"
                  f"{viol_str}")

    # ── Plots ──────────────────────────────────────────────────────────────
    _comparison_bar(summary, out_dir / "all_methods_comparison.png")
    _pareto_scatter(summary, out_dir)
    _episode_violation_bar(summary, out_dir)
    _kcbf_diagnostics(summary, out_dir)

    # ── Per-env training curves ────────────────────────────────────────────
    if not a.skip_training_curves:
        env_names = sorted(summary["env"].unique())
        for env_name in env_names:
            _training_curves(
                baseline_root, kcbf_root, env_name,
                out_dir / f"training_curves_{env_name}.png",
            )

    # ── LaTeX table ────────────────────────────────────────────────────────
    _latex_table(summary, out_dir / "latex_table.txt")

    print(f"\nAll outputs in {out_dir}/")


if __name__ == "__main__":
    main()
