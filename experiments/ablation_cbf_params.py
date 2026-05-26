"""Ablation study: η (CBF decay) and λ_slack (QP slack penalty) for Safety Gym KCBF-SAC.

Grid sweeps η ∈ {0.3, 0.5, 0.7, 0.9} × λ_slack ∈ {100, 1000, 10000} per env.
Each config runs one seed with total_steps controlled by --total_steps (default 300k).
Completed jobs are skipped automatically (idempotent).

Outputs:
    {log_root}/ablation_results.csv          — per-config metrics
    {log_root}/eta_lam_heatmap.png           — 2D heatmap: η × λ_slack for each metric
    {log_root}/safety_efficiency_tradeoff.png — pareto curve: return vs violation_rate

Usage:
    # Quick ablation (300k steps, safety_walker only)
    python experiments/ablation_cbf_params.py \\
      --envs safety_walker \\
      --model_dir logs/models \\
      --log_root logs/ablation \\
      --total_steps 300000

    # Full ablation (1M steps, both Safety Gym envs)
    python experiments/ablation_cbf_params.py \\
      --envs safety_walker safety_halfcheetah \\
      --model_dir logs/models \\
      --log_root logs/ablation \\
      --total_steps 1000000 --workers 2

    # Custom grid
    python experiments/ablation_cbf_params.py \\
      --envs safety_walker \\
      --eta 0.5 0.7 0.9 --lam_slack 500 1000 5000 10000 \\
      --model_dir logs/models --log_root logs/ablation
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_CFG = _ROOT / "robust_koopman_cbf_rl" / "configs"

ENVS: dict[str, str] = {
    "safety_halfcheetah": str(_CFG / "env_safety_halfcheetah.yaml"),
    "safety_walker":      str(_CFG / "env_safety_walker.yaml"),
}

_BASE_SAC_CFG = str(_CFG / "sac_kcbf_safety_gym.yaml")

# Base config YAML template — overrides eta and lam_slack
_CFG_TEMPLATE = """\
# Auto-generated ablation config — do not edit.
total_steps: {total_steps}
warmup_steps: 10000
batch_size: 256
buffer_capacity: 1000000
gamma: 0.99
tau: 0.005
lr: 3.0e-4
alpha: 0.2
lam_h: 1.0
eta: {eta}
lam_slack: {lam_slack}
"""


def _job_label(env_name: str, eta: float, lam_slack: float) -> str:
    return f"{env_name}/eta{eta}_lam{int(lam_slack)}"


def _run_job(
    env_name: str,
    eta: float,
    lam_slack: float,
    model_path: str,
    log_dir: Path,
    python: str,
    total_steps: int,
) -> tuple[str, str]:
    label = _job_label(env_name, eta, lam_slack)
    done = log_dir / "eval_final.csv"
    if done.exists():
        print(f"[SKIP] {label}")
        return label, "skipped"
    if not Path(model_path).exists():
        print(f"[SKIP] {label} — Koopman model missing: {model_path}")
        return label, "no_model"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Write temporary config with overridden eta/lam_slack
    cfg_content = _CFG_TEMPLATE.format(
        eta=eta, lam_slack=lam_slack, total_steps=total_steps
    )
    tmp_cfg = log_dir / "sac_kcbf_ablation.yaml"
    tmp_cfg.write_text(cfg_content)

    cmd = [
        python, "-m", "robust_koopman_cbf_rl.train.train_sac_kcbf",
        "--env_cfg", ENVS[env_name],
        "--sac_cfg", str(tmp_cfg),
        "--model", model_path,
        "--log_dir", str(log_dir),
        "--total_steps", str(total_steps),
        "--seed", "0",
    ]
    log_file = log_dir / "stdout.log"
    print(f"[RUN]  {label}  (eta={eta}, lam_slack={lam_slack}, steps={total_steps})")
    with open(log_file, "w") as f:
        rc = subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT, text=True, cwd=str(_ROOT)
        ).returncode
    if rc == 0:
        print(f"[DONE] {label}")
        return label, "ok"
    print(f"[FAIL] {label}  (rc={rc}, see {log_file})")
    return label, f"failed(rc={rc})"


def _load_metrics(log_dir: Path) -> dict | None:
    f = log_dir / "eval_final.csv"
    if not f.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(f)
        out = {}
        for col in ["return", "total_cost", "violation_rate", "episode_violation",
                    "min_h_value", "intervention_rate"]:
            vals = df[col].dropna().values if col in df.columns else np.array([])
            out[col] = float(np.mean(vals)) if len(vals) > 0 else float("nan")
        return out
    except Exception as exc:
        warnings.warn(f"Failed to load {f}: {exc}")
        return None


def _plot_heatmaps(
    df,
    env_name: str,
    eta_list: list[float],
    lam_list: list[float],
    out_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        warnings.warn("matplotlib not available — skipping plots")
        return

    sub = df[df["env"] == env_name].copy()
    if sub.empty:
        return

    metrics = [
        ("violation_rate", "Violation Rate", "Reds", False),
        ("return",         "Return",         "Greens", False),
        ("intervention_rate", "Intervention Rate", "Blues", False),
    ]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))

    for ax, (col, title, cmap, logscale) in zip(axes, metrics):
        grid = np.full((len(eta_list), len(lam_list)), np.nan)
        for i, eta in enumerate(eta_list):
            for j, lam in enumerate(lam_list):
                row = sub[(sub["eta"] == eta) & (sub["lam_slack"] == lam)]
                if not row.empty and col in row.columns:
                    grid[i, j] = float(row[col].values[0])

        im = ax.imshow(grid, cmap=cmap, aspect="auto",
                       norm=LogNorm() if logscale else None)
        ax.set_xticks(range(len(lam_list)))
        ax.set_xticklabels([f"{int(l)}" for l in lam_list], rotation=30)
        ax.set_yticks(range(len(eta_list)))
        ax.set_yticklabels([str(e) for e in eta_list])
        ax.set_xlabel("λ_slack")
        ax.set_ylabel("η")
        ax.set_title(f"{title}\n({env_name})")
        plt.colorbar(im, ax=ax, shrink=0.8)
        # Annotate cells
        for i in range(len(eta_list)):
            for j in range(len(lam_list)):
                v = grid[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=8, color="black")

    fig.suptitle(f"η × λ_slack Ablation — {env_name}", fontsize=12)
    plt.tight_layout()
    out = out_dir / f"heatmap_{env_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def _plot_tradeoff(df, env_name: str, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        return

    sub = df[df["env"] == env_name].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    eta_vals = sorted(sub["eta"].unique())
    colors = cm.viridis(np.linspace(0.2, 0.9, len(eta_vals)))
    markers = ["o", "s", "^", "D"]

    for i, (eta, color) in enumerate(zip(eta_vals, colors)):
        grp = sub[sub["eta"] == eta].sort_values("lam_slack")
        ax.plot(grp["violation_rate"], grp["return"],
                color=color, marker=markers[i % len(markers)],
                linewidth=1.5, markersize=8, label=f"η={eta}")
        for _, row in grp.iterrows():
            ax.annotate(
                f"λ={int(row['lam_slack'])}",
                (row["violation_rate"], row["return"]),
                textcoords="offset points", xytext=(5, 3), fontsize=7,
            )

    ax.set_xlabel("Violation Rate (↓ better)")
    ax.set_ylabel("Return (↑ better)")
    ax.set_title(f"Safety–Efficiency Tradeoff: {env_name}")
    ax.legend(title="CBF decay η")
    ax.grid(True, alpha=0.3)

    out = out_dir / f"tradeoff_{env_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="η × λ_slack ablation for Safety Gym KCBF-SAC.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--envs", nargs="+", default=["safety_walker"],
                    choices=list(ENVS))
    ap.add_argument("--eta", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--lam_slack", type=float, nargs="+", default=[100.0, 1000.0, 10000.0])
    ap.add_argument("--model_dir", default=None,
                    help="Root dir containing <env_name>/koopman.npz. "
                         "If it directly contains koopman.npz, used for all envs. "
                         "Use --model_dirs for per-env paths.")
    ap.add_argument("--model_dirs", nargs="+", metavar="ENV=PATH", default=[],
                    help="Per-env model dirs, e.g. safety_walker=logs/models/safety_walker_tuned "
                         "safety_halfcheetah=logs/models/safety_halfcheetah. "
                         "Overrides --model_dir for specified envs.")
    ap.add_argument("--log_root", required=True,
                    help="Root directory for ablation run logs")
    ap.add_argument("--total_steps", type=int, default=300_000,
                    help="Training steps per config (default: 300k for speed)")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel workers (default: 1)")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    if a.model_dir is None and not a.model_dirs:
        ap.error("Provide --model_dir or --model_dirs.")

    # Build per-env model path map from --model_dirs overrides
    per_env_dir: dict[str, Path] = {}
    for spec in a.model_dirs:
        env_name, _, path = spec.partition("=")
        per_env_dir[env_name.strip()] = Path(path.strip())

    log_root = Path(a.log_root)
    model_dir = Path(a.model_dir) if a.model_dir else None
    log_root.mkdir(parents=True, exist_ok=True)

    n_total = len(a.envs) * len(a.eta) * len(a.lam_slack)
    print(f"CBF parameter ablation: {n_total} configs")
    print(f"  envs:      {a.envs}")
    print(f"  eta:       {a.eta}")
    print(f"  lam_slack: {a.lam_slack}")
    print(f"  steps:     {a.total_steps}")
    print(f"  workers:   {a.workers}\n")

    jobs = []
    for env_name in a.envs:
        # Resolve model path with priority: --model_dirs > --model_dir direct > --model_dir/env
        if env_name in per_env_dir:
            d = per_env_dir[env_name]
        elif model_dir is not None:
            d = model_dir
        else:
            ap.error(f"No model dir specified for env '{env_name}'. Use --model_dir or --model_dirs.")
        if (d / "koopman.npz").exists():
            model_path = str(d / "koopman.npz")
        else:
            model_path = str(d / env_name / "koopman.npz")
        for eta, lam in product(a.eta, a.lam_slack):
            log_dir = log_root / env_name / f"eta{eta}_lam{int(lam)}"
            jobs.append((env_name, eta, lam, model_path, log_dir, a.python, a.total_steps))

    results: dict[str, str] = {}
    if a.workers == 1:
        for job in jobs:
            label, status = _run_job(*job)
            results[label] = status
    else:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures = {pool.submit(_run_job, *job): job for job in jobs}
            for fut in as_completed(futures):
                label, status = fut.result()
                results[label] = status

    ok = sum(1 for s in results.values() if s == "ok")
    skipped = sum(1 for s in results.values() if s == "skipped")
    failed = [lbl for lbl, s in results.items() if s.startswith("failed")]
    print(f"\n=== Ablation: {ok} done, {skipped} skipped, {len(failed)} failed ===")
    if failed:
        print("Failed:")
        for lbl in failed:
            print(f"  {lbl}: {results[lbl]}")

    # ── Collect results and plot ───────────────────────────────────────────────
    import pandas as pd
    rows = []
    for env_name in a.envs:
        for eta, lam in product(a.eta, a.lam_slack):
            log_dir = log_root / env_name / f"eta{eta}_lam{int(lam)}"
            m = _load_metrics(log_dir)
            if m is None:
                continue
            rows.append({
                "env":             env_name,
                "eta":             eta,
                "lam_slack":       lam,
                "return":          m.get("return", float("nan")),
                "violation_rate":  m.get("violation_rate", float("nan")),
                "total_cost":      m.get("total_cost", float("nan")),
                "intervention_rate": m.get("intervention_rate", float("nan")),
                "min_h_value":     m.get("min_h_value", float("nan")),
            })

    if not rows:
        print("No completed runs found — nothing to plot yet.")
        return

    df = pd.DataFrame(rows)
    out_csv = log_root / "ablation_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nResults → {out_csv}")

    # Print ranked table per env
    for env_name in a.envs:
        sub = df[df["env"] == env_name].sort_values("violation_rate")
        print(f"\n{'─'*70}")
        print(f"Ablation results: {env_name}  (sorted by violation_rate ↑)")
        print(f"{'─'*70}")
        cols = ["eta", "lam_slack", "return", "violation_rate",
                "intervention_rate", "min_h_value"]
        present = [c for c in cols if c in sub.columns]
        print(sub[present].to_string(index=False))
        best = sub.iloc[0]
        print(f"\n★ Best (lowest violation):  η={best['eta']}  λ_slack={int(best['lam_slack'])}")
        print(f"  violation_rate={best['violation_rate']:.4f}  return={best['return']:.2f}")

    # Plots
    print("\n--- Generating plots ---")
    for env_name in a.envs:
        _plot_heatmaps(df, env_name, a.eta, a.lam_slack, log_root)
        _plot_tradeoff(df, env_name, log_root)

    print("\nAblation complete.")


if __name__ == "__main__":
    main()
