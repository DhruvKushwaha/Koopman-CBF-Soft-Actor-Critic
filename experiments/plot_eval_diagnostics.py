"""Diagnostic plot orchestration: constraint eval + training curves + filter timing.

Generates all diagnostic figures for an experiment sweep:
  1. Per-env multi-seed training curves (mean±std bands)  [plot_training_multirun]
  2. Filter timing benchmark                              [benchmark_filters]
  3. Constraint eval plots from saved .npz trace files   [plot_constraint_eval]
     (requires eval_constraint_trace.npz files, generated separately)

Quick usage (no constraint traces needed):
    python experiments/plot_eval_diagnostics.py \\
      --baseline_root logs/sweep \\
      --kcbf_root     logs/kcbf_sweep \\
      --out_dir       logs/diagnostics

Run timing benchmark only:
    python experiments/plot_eval_diagnostics.py \\
      --out_dir logs/diagnostics \\
      --only_timing

Run constraint plots only (needs --constraint_trace_dir):
    python experiments/plot_eval_diagnostics.py \\
      --constraint_trace_dir logs/traces \\
      --out_dir logs/diagnostics \\
      --only_constraint

Generating constraint traces:
    The eval_constraint_trace.npz files must be generated manually using
    eval_constraint_trace.eval_constraint_trace() after loading trained agents,
    since the trace needs live env + agent + model objects.
    See robust_koopman_cbf_rl/eval/eval_constraint_trace.py for the API.
"""
from __future__ import annotations
import argparse
import warnings
from pathlib import Path

_ROOT = Path(__file__).parent.parent

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

_TRAIN_CSV = {
    "sac":            "sac_baseline.csv",
    "ppo":            "ppo_baseline.csv",
    "sac_penalty":    "sac_penalty.csv",
    "sac_lagrangian": "sac_lagrangian.csv",
    "kcbf_sac":       "sac_kcbf.csv",
    "kcbf_ppo":       "ppo_kcbf.csv",
    "sac_run1":       "sac_kcbf.csv",
    "ppo_run1":       "ppo_kcbf.csv",
}


def _discover_envs(*roots: Path | None) -> set[str]:
    envs: set[str] = set()
    for root in roots:
        if root and root.exists():
            for d in root.iterdir():
                if d.is_dir():
                    envs.add(d.name)
    return envs


def _build_csv_groups(baseline_root: Path | None, kcbf_root: Path | None,
                      env_name: str) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for root in [baseline_root, kcbf_root]:
        if not root or not (root / env_name).exists():
            continue
        env_dir = root / env_name
        for bl_dir in sorted(env_dir.iterdir()):
            if not bl_dir.is_dir():
                continue
            bl_name = bl_dir.name
            csv_name = _TRAIN_CSV.get(bl_name)
            label = _LABEL_MAP.get(bl_name)
            if not csv_name or not label:
                continue
            seed_dirs = sorted(
                d for d in bl_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")
            )
            found = [str(sd / csv_name) for sd in seed_dirs if (sd / csv_name).exists()]
            if found:
                groups.setdefault(label, []).extend(found)
    return groups


def run_training_curves(baseline_root: Path | None, kcbf_root: Path | None,
                        out_dir: Path) -> None:
    from robust_koopman_cbf_rl.plots.plot_training_multirun import plot_training_multirun
    envs = _discover_envs(baseline_root, kcbf_root)
    if not envs:
        print("  No environments found for training curves.")
        return
    for env_name in sorted(envs):
        groups = _build_csv_groups(baseline_root, kcbf_root, env_name)
        if not groups:
            print(f"  No training CSVs for {env_name}")
            continue
        out_path = out_dir / f"training_{env_name}.png"
        try:
            plot_training_multirun(groups, str(out_path))
        except Exception as e:
            warnings.warn(f"Training curves failed for {env_name}: {e}")


def run_filter_benchmark(out_dir: Path, n_trials: int = 1000, warmup: int = 50) -> None:
    import sys; sys.path.insert(0, str(_ROOT))
    try:
        from robust_koopman_cbf_rl.eval.benchmark_filters import run_full_benchmark
        from robust_koopman_cbf_rl.plots.plot_filter_timing import plot_filter_timing
        import pandas as pd
        print("  Running filter benchmark...")
        df = run_full_benchmark(n_trials=n_trials, warmup=warmup)
        csv_path = out_dir / "filter_timing_raw.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Raw timings → {csv_path}")
        plot_filter_timing(df, str(out_dir / "filter_timing.png"))
    except Exception as e:
        warnings.warn(f"Filter benchmark failed: {e}")
        print(f"  FAILED: {e}")


def run_constraint_plots(trace_dir: Path, out_dir: Path) -> None:
    """Load pre-generated .npz trace files and produce constraint eval plots per env."""
    import numpy as np
    from robust_koopman_cbf_rl.plots.plot_constraint_eval import plot_constraint_eval

    # Expected layout: {trace_dir}/{env_name}/{method_label}.npz
    if not trace_dir.exists():
        print(f"  Trace directory not found: {trace_dir}")
        return

    env_names = [d.name for d in trace_dir.iterdir() if d.is_dir()]
    if not env_names:
        print(f"  No env subdirectories in {trace_dir}")
        return

    for env_name in sorted(env_names):
        env_trace_dir = trace_dir / env_name
        npz_files = sorted(env_trace_dir.glob("*.npz"))
        if not npz_files:
            continue
        traces_dict = {}
        for npz in npz_files:
            label = npz.stem  # filename without .npz = method label
            try:
                data = dict(np.load(npz))
                traces_dict[label] = data
            except Exception as e:
                warnings.warn(f"Could not load {npz}: {e}")
        if not traces_dict:
            continue
        out_path = out_dir / f"constraint_eval_{env_name}.png"
        try:
            plot_constraint_eval(traces_dict, str(out_path))
        except Exception as e:
            warnings.warn(f"Constraint plot failed for {env_name}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate all diagnostic evaluation plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--baseline_root", default=None,
                    help="Sweep root from run_sweep.py")
    ap.add_argument("--kcbf_root", default=None,
                    help="Sweep root from run_kcbf_sweep.py")
    ap.add_argument("--constraint_trace_dir", default=None,
                    help="Directory of eval_constraint_trace.npz files "
                         "(layout: {dir}/{env_name}/{method_label}.npz)")
    ap.add_argument("--out_dir", required=True, help="Output directory for all plots")
    ap.add_argument("--n_timing_trials", type=int, default=1000)
    ap.add_argument("--only_timing", action="store_true",
                    help="Run only the filter timing benchmark")
    ap.add_argument("--only_constraint", action="store_true",
                    help="Run only the constraint eval plots")
    ap.add_argument("--skip_timing", action="store_true",
                    help="Skip filter timing benchmark")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_root = Path(a.baseline_root) if a.baseline_root else None
    kcbf_root = Path(a.kcbf_root) if a.kcbf_root else None

    if a.only_timing:
        print("=== Filter Timing Benchmark ===")
        run_filter_benchmark(out_dir, n_trials=a.n_timing_trials)
        return

    if a.only_constraint:
        if not a.constraint_trace_dir:
            ap.error("--constraint_trace_dir required with --only_constraint")
        print("=== Constraint Eval Plots ===")
        run_constraint_plots(Path(a.constraint_trace_dir), out_dir)
        return

    # ── Full diagnostic run ────────────────────────────────────────────────
    print("=== 1/3  Multi-seed training curves ===")
    if baseline_root or kcbf_root:
        run_training_curves(baseline_root, kcbf_root, out_dir)
    else:
        print("  (skipped — no --baseline_root or --kcbf_root)")

    if not a.skip_timing:
        print("\n=== 2/3  Filter timing benchmark ===")
        run_filter_benchmark(out_dir, n_trials=a.n_timing_trials)

    print("\n=== 3/3  Constraint eval plots ===")
    if a.constraint_trace_dir:
        run_constraint_plots(Path(a.constraint_trace_dir), out_dir)
    else:
        print("  (skipped — no --constraint_trace_dir provided)")
        print("  Generate traces first with eval_constraint_trace.eval_constraint_trace()")

    print(f"\nDiagnostics complete → {out_dir}/")


if __name__ == "__main__":
    main()
