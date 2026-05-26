"""Analyze and plot Quadrotor 2D results (stabilization + tracking).

Loads all eval_final.csv files for quadrotor2d_stab and quadrotor2d_track from
both baseline_sweep and kcbf_sweep, then produces:
  1. Bar chart: return and violation_rate per method (stab and track side-by-side)
  2. Safety-efficiency scatter: return vs violation_rate
  3. Koopman tuning surface: rho_95 vs (n_rbf, steps)
  4. Summary table printed to stdout

Usage:
    python experiments/analyze_quadrotor.py \\
        --baseline_root logs/baseline_sweep \\
        --kcbf_root     logs/kcbf_sweep \\
        --koopman_csv   logs/koopman_tuning/quadrotor_results.csv \\
        --out_dir       logs/results_quadrotor
"""
from __future__ import annotations
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent


# ── helpers ────────────────────────────────────────────────────────────────

def _load_seeded(log_root: Path, env: str, method: str, seeds=(1, 2, 3)) -> list[dict]:
    rows = []
    for s in seeds:
        f = log_root / env / method / f"seed_{s}" / "eval_final.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        rows.append({
            "return":          float(df["return"].mean()),
            "violation_rate":  float(df["violation_rate"].mean()),
            "total_cost":      float(df["total_cost"].mean()),
            "episode_violation": float(df["episode_violation"].mean()),
        })
    return rows


def _load_classical(log_root: Path, env: str, method: str) -> list[dict] | None:
    f = log_root / env / method / "eval_final.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    return [{
        "return":           float(df["return"].mean()),
        "violation_rate":   float(df["violation_rate"].mean()),
        "total_cost":       float(df["total_cost"].mean()),
        "episode_violation": float(df["episode_violation"].mean()),
    }]


def _stats(rows: list[dict], key: str) -> tuple[float, float]:
    vals = [r[key] for r in rows]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def _collect(baseline_root: Path, kcbf_root: Path,
             envs=("quadrotor2d_stab", "quadrotor2d_track")) -> pd.DataFrame:
    records = []
    methods_bl = [
        ("sac",           "SAC",           baseline_root),
        ("sac_penalty",   "SAC-Penalty",   baseline_root),
        ("sac_lagrangian","SAC-Lagrangian", baseline_root),
    ]
    methods_kcbf = [
        ("kcbf_sac", "KCBF-SAC", kcbf_root),
    ]
    classical = [("lqr", "LQR"), ("pid", "PID")]

    for env in envs:
        for method, label, root in methods_bl + methods_kcbf:
            rows = _load_seeded(root, env, method)
            if not rows:
                continue
            ret_m, ret_s = _stats(rows, "return")
            viol_m, viol_s = _stats(rows, "violation_rate")
            records.append({"env": env, "method": label,
                             "return_mean": ret_m, "return_std": ret_s,
                             "viol_mean": viol_m, "viol_std": viol_s,
                             "n_seeds": len(rows)})

        for method, label in classical:
            rows = _load_classical(baseline_root, env, method)
            if not rows:
                continue
            records.append({"env": env, "method": label,
                             "return_mean": rows[0]["return"],
                             "return_std": 0.0,
                             "viol_mean": rows[0]["violation_rate"],
                             "viol_std": 0.0,
                             "n_seeds": 1})

    return pd.DataFrame(records)


# ── plots ───────────────────────────────────────────────────────────────────

METHOD_ORDER = ["LQR", "PID", "SAC", "SAC-Penalty", "SAC-Lagrangian", "KCBF-SAC"]
METHOD_COLORS = {
    "LQR":           "#888888",
    "PID":           "#bbbbbb",
    "SAC":           "#4878d0",
    "SAC-Penalty":   "#ee854a",
    "SAC-Lagrangian":"#6acc65",
    "KCBF-SAC":      "#d65f5f",
}


def _plot_bar(df: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not available")
        return

    envs = ["quadrotor2d_stab", "quadrotor2d_track"]
    env_labels = {"quadrotor2d_stab": "Stab", "quadrotor2d_track": "Track"}
    metrics = [("return_mean", "return_std", "Episode Return (↑)"),
               ("viol_mean",   "viol_std",   "Violation Rate (↓)")]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 7))

    for ax, (m_mean, m_std, ylabel) in zip(axes, metrics):
        x = np.arange(len(envs))
        methods_present = [m for m in METHOD_ORDER
                           if m in df["method"].values]
        n_m = len(methods_present)
        width = 0.8 / n_m

        for i, method in enumerate(methods_present):
            vals, errs = [], []
            for env in envs:
                row = df[(df["env"] == env) & (df["method"] == method)]
                if row.empty:
                    vals.append(float("nan")); errs.append(0.0)
                else:
                    vals.append(float(row[m_mean].values[0]))
                    errs.append(float(row[m_std].values[0]))
            offset = (i - n_m / 2 + 0.5) * width
            ax.bar(x + offset, vals, width * 0.9, yerr=errs, capsize=3,
                   label=method, color=METHOD_COLORS.get(method, "gray"), alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([env_labels[e] for e in envs])
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(axis="y", alpha=0.3)
        if "viol" in m_mean:
            ax.set_ylim(bottom=0)
        ax.set_title(f"Quadrotor 2D — {ylabel}")

    plt.tight_layout()
    out = out_dir / "quadrotor_bar_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def _plot_tradeoff(df: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    envs = ["quadrotor2d_stab", "quadrotor2d_track"]
    env_labels = {"quadrotor2d_stab": "Stab", "quadrotor2d_track": "Track"}
    markers = {"quadrotor2d_stab": "o", "quadrotor2d_track": "s"}

    fig, ax = plt.subplots(figsize=(8, 5))

    for env in envs:
        sub = df[df["env"] == env]
        for _, row in sub.iterrows():
            method = row["method"]
            color = METHOD_COLORS.get(method, "gray")
            ax.scatter(row["viol_mean"], row["return_mean"],
                       marker=markers[env], color=color, s=100, zorder=5,
                       label=f"{method} ({env_labels[env]})")
            ax.errorbar(row["viol_mean"], row["return_mean"],
                        xerr=row["viol_std"], yerr=row["return_std"],
                        fmt="none", color=color, alpha=0.5, capsize=3)
            ax.annotate(f"{method}\n({env_labels[env]})",
                        (row["viol_mean"], row["return_mean"]),
                        textcoords="offset points", xytext=(5, 3), fontsize=7)

    ax.set_xlabel("Violation Rate (↓ better)")
    ax.set_ylabel("Episode Return (↑ better)")
    ax.set_title("Quadrotor 2D — Safety–Efficiency Tradeoff")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "quadrotor_tradeoff.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def _plot_koopman_tuning(csv_path: Path, out_dir: Path) -> None:
    if not csv_path.exists():
        warnings.warn(f"Koopman tuning CSV not found: {csv_path}")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        return

    df = pd.read_csv(csv_path)
    # Filter to quadrotor (stab, since both are identical)
    sub = df[df["env"].str.contains("quadrotor") & (df["env"].str.contains("stab"))].copy()
    if sub.empty:
        warnings.warn("No quadrotor tuning data found in CSV")
        return

    n_rbf_vals = sorted(sub["n_rbf"].unique())
    steps_col = "steps" if "steps" in sub.columns else "n_steps"
    steps_vals = sorted(sub[steps_col].unique())

    # compute rho_ratio if not present
    if "rho_ratio" not in sub.columns and "rho_max" in sub.columns and "rho_95" in sub.columns:
        sub = sub.copy()
        sub["rho_ratio"] = sub["rho_max"] / sub["rho_95"].clip(lower=1e-9)

    metrics = [
        ("rho_95",   "rho_95 (↓ better)",   "Blues_r"),
        ("rho_max",  "rho_max (↓ better)",   "Reds_r"),
        ("rho_ratio","rho_ratio (↓ better)", "Purples_r"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))

    for ax, (col, title, cmap) in zip(axes, metrics):
        if col not in sub.columns:
            continue
        grid = np.full((len(n_rbf_vals), len(steps_vals)), np.nan)
        for i, n in enumerate(n_rbf_vals):
            for j, s in enumerate(steps_vals):
                row = sub[(sub["n_rbf"] == n) & (sub[steps_col] == s)]
                if not row.empty:
                    grid[i, j] = float(row[col].values[0])

        im = ax.imshow(grid, cmap=cmap, aspect="auto",
                       norm=LogNorm(vmin=np.nanmin(grid), vmax=np.nanmax(grid))
                       if np.nanmin(grid) > 0 else None)
        ax.set_xticks(range(len(steps_vals)))
        ax.set_xticklabels([f"{int(s//1000)}k" for s in steps_vals], rotation=30)
        ax.set_yticks(range(len(n_rbf_vals)))
        ax.set_yticklabels([str(n) for n in n_rbf_vals])
        ax.set_xlabel("Collection Steps")
        ax.set_ylabel("n_rbf")
        ax.set_title(f"{title}\n(Quadrotor Stab)")
        plt.colorbar(im, ax=ax, shrink=0.8)
        for i in range(len(n_rbf_vals)):
            for j in range(len(steps_vals)):
                v = grid[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.4f}", ha="center", va="center",
                            fontsize=7, color="black")

    fig.suptitle("Koopman Tuning — Quadrotor 2D", fontsize=12)
    plt.tight_layout()
    out = out_dir / "quadrotor_koopman_tuning.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze and plot Quadrotor 2D results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--baseline_root", default="logs/baseline_sweep",
                    help="Root dir of baseline sweep (default: logs/baseline_sweep)")
    ap.add_argument("--kcbf_root", default="logs/kcbf_sweep",
                    help="Root dir of KCBF sweep (default: logs/kcbf_sweep)")
    ap.add_argument("--koopman_csv",
                    default="logs/koopman_tuning/quadrotor_results.csv",
                    help="Koopman tuning CSV (default: logs/koopman_tuning/quadrotor_results.csv)")
    ap.add_argument("--out_dir", default="logs/results_quadrotor",
                    help="Output directory for plots and CSV (default: logs/results_quadrotor)")
    a = ap.parse_args()

    baseline_root = _ROOT / a.baseline_root
    kcbf_root     = _ROOT / a.kcbf_root
    koopman_csv   = _ROOT / a.koopman_csv
    out_dir       = _ROOT / a.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting results...")
    df = _collect(baseline_root, kcbf_root)
    if df.empty:
        print("No results found. Check --baseline_root and --kcbf_root paths.")
        return

    csv_out = out_dir / "quadrotor_summary.csv"
    df.to_csv(csv_out, index=False)
    print(f"Summary → {csv_out}\n")

    # ── Print table ──────────────────────────────────────────────────────
    for env in ("quadrotor2d_stab", "quadrotor2d_track"):
        sub = df[df["env"] == env].copy()
        if sub.empty:
            continue
        available = set(sub["method"].values)
        sub = sub.set_index("method").reindex([m for m in METHOD_ORDER if m in available])
        print(f"\n{'─'*70}")
        print(f"  {env.replace('_', ' ').title()}")
        print(f"{'─'*70}")
        print(f"{'Method':<18} {'Return':>10} {'±':>5} {'Viol%':>8} {'±':>5} {'Seeds':>5}")
        print(f"{'─'*70}")
        for method, row in sub.iterrows():
            if pd.isna(row["return_mean"]):
                continue
            viol_pct = row["viol_mean"] * 100
            viol_std = row["viol_std"] * 100
            print(f"{method:<18} {row['return_mean']:>10.2f} {row['return_std']:>5.2f}"
                  f"  {viol_pct:>7.3f}% {viol_std:>5.3f}  {int(row['n_seeds']):>4}x")

    # ── Find best KCBF vs best baseline ──────────────────────────────────
    print(f"\n{'═'*70}")
    print("  KCBF-SAC vs baselines (safety gain)")
    print(f"{'═'*70}")
    for env in ("quadrotor2d_stab", "quadrotor2d_track"):
        sub = df[df["env"] == env]
        kcbf = sub[sub["method"] == "KCBF-SAC"]
        sac  = sub[sub["method"] == "SAC"]
        if kcbf.empty or sac.empty:
            continue
        kv = float(kcbf["viol_mean"].values[0])
        sv = float(sac["viol_mean"].values[0])
        kr = float(kcbf["return_mean"].values[0])
        sr = float(sac["return_mean"].values[0])
        gain = (sv - kv) / max(sv, 1e-9) * 100
        cost = (sr - kr) / max(abs(sr), 1e-9) * 100
        print(f"  {env}: viol {sv:.4f}→{kv:.4f} ({gain:+.1f}%), "
              f"return {sr:.1f}→{kr:.1f} ({cost:+.1f}%)")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n--- Generating plots ---")
    _plot_bar(df, out_dir)
    _plot_tradeoff(df, out_dir)
    _plot_koopman_tuning(koopman_csv, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
