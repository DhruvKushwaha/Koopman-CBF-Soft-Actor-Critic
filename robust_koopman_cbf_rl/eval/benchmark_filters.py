"""Per-iteration wall-clock timing benchmark for KCBF-QP, Physical-CBF, and Null filters.

Generates a DataFrame of timing measurements and saves comparison plots.

Design notes:
  - KCBFQPFilter: OSQP QP solve per step; cost scales with u_dim (Gram matrix is u_dim × u_dim).
  - PhysicalCBFFilter: OSQP QP + u_dim serial finite-difference dynamics calls.
    Timing shown uses a trivial numpy dynamics function; real MuJoCo/CasADi dynamics
    will be substantially slower (noted in plot title).
  - NullFilter: Pure numpy clip — baseline for overhead measurement.

Usage:
    python -m robust_koopman_cbf_rl.eval.benchmark_filters --out_dir logs/timing
    python -m robust_koopman_cbf_rl.eval.benchmark_filters --n_trials 2000 --out_dir logs/timing
"""
from __future__ import annotations
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# ── Synthetic filter construction ────────────────────────────────────────────

def _make_koopman_filter(z_dim: int, u_dim: int, eta: float = 0.5,
                          lam_slack: float = 1e3) -> object:
    """Synthetic KCBFQPFilter with random (stable) A, random B, linear barrier."""
    from robust_koopman_cbf_rl.cbf.qp_filter import KCBFQPFilter
    from robust_koopman_cbf_rl.cbf.robust_margin import RobustMargin

    rng = np.random.default_rng(0)
    # Stable A: scale random matrix to have spectral radius < 1
    A_raw = rng.standard_normal((z_dim, z_dim)) * 0.05
    A = A_raw - np.eye(z_dim) * 0.1   # ensure negative diagonal dominance
    B = rng.standard_normal((z_dim, u_dim)) * 0.1
    c = np.zeros(z_dim); c[-1] = 1.0  # barrier: last lifted dimension
    d = 0.5

    class _Barrier:
        def lifted_barrier_coeffs(self, **kwargs):
            return c, d

    class _Model:
        pass

    model = _Model()
    model.A = A
    model.B = B
    rm = RobustMargin(alpha=0.95, mode="global")
    rm.update(np.zeros(20))
    return KCBFQPFilter(
        model, _Barrier(), rm, eta=eta, lam_slack=lam_slack,
        u_min=np.full(u_dim, -1.0), u_max=np.full(u_dim, 1.0),
    )


def _make_physical_filter(state_dim: int, u_dim: int,
                           eta: float = 0.5) -> object:
    """Synthetic PhysicalCBFFilter with trivial linear dynamics f(x,u) = x + 0.01 Bu."""
    from robust_koopman_cbf_rl.baselines.physical_cbf_qp import PhysicalCBFFilter

    rng = np.random.default_rng(1)
    B = rng.standard_normal((state_dim, u_dim)) * 0.01
    c = np.zeros(state_dim); c[-1] = 1.0

    def dynamics(x, u):
        return x + B @ u

    return PhysicalCBFFilter(
        dynamics=dynamics,
        dim_state=state_dim,
        dim_action=u_dim,
        h_coeffs=(c, 0.5),
        eta=eta,
        u_min=np.full(u_dim, -1.0),
        u_max=np.full(u_dim, 1.0),
    )


def _make_null_filter(u_dim: int) -> object:
    """NullFilter (clip-only) for overhead baseline."""
    from robust_koopman_cbf_rl.cbf.null_filter import NullFilter
    return NullFilter(u_min=np.full(u_dim, -1.0), u_max=np.full(u_dim, 1.0))


# ── Timing routine ────────────────────────────────────────────────────────────

def _time_filter(flt, z: np.ndarray, u: np.ndarray,
                 n_trials: int, warmup: int) -> np.ndarray:
    """Return per-call wall-clock times in milliseconds (float64 array, length n_trials)."""
    # Warmup to avoid JIT / OSQP setup artifacts
    for _ in range(warmup):
        try:
            flt.project(z, u)
        except Exception:
            pass
    times = np.empty(n_trials)
    for k in range(n_trials):
        t0 = time.perf_counter()
        try:
            flt.project(z, u)
        except Exception:
            pass
        times[k] = (time.perf_counter() - t0) * 1e3  # ms
    return times


# ── Configuration table ───────────────────────────────────────────────────────

# Realistic (env, z_dim, state_dim, u_dim) tuples
ENV_CONFIGS = [
    ("CartPole",      68,  4,  1),
    ("Quadrotor2D",   70,  6,  2),
    ("HalfCheetah",   81, 17,  6),
    ("Walker2D",      81, 17,  6),
    ("Large (u=12)",  92, 28, 12),
]


def run_full_benchmark(
    n_trials: int = 1000,
    warmup: int = 50,
    filter_names: tuple[str, ...] = ("NullFilter", "KCBFQPFilter", "PhysicalCBFFilter"),
) -> pd.DataFrame:
    """Run timing benchmark across all env configs and filter types.

    Returns a DataFrame with columns:
        env, filter, u_dim, z_dim, trial_ms (one row per trial)
    """
    rows = []
    rng = np.random.default_rng(42)

    for env_name, z_dim, state_dim, u_dim in ENV_CONFIGS:
        z = rng.standard_normal(z_dim)
        u = rng.uniform(-0.5, 0.5, u_dim)
        x = rng.standard_normal(state_dim)

        filters: dict[str, object] = {}
        if "NullFilter" in filter_names:
            filters["NullFilter"] = _make_null_filter(u_dim)
        if "KCBFQPFilter" in filter_names:
            try:
                filters["KCBFQPFilter"] = _make_koopman_filter(z_dim, u_dim)
            except Exception as e:
                warnings.warn(f"KCBFQPFilter construction failed for {env_name}: {e}")
        if "PhysicalCBFFilter" in filter_names:
            try:
                filters["PhysicalCBFFilter"] = _make_physical_filter(state_dim, u_dim)
            except Exception as e:
                warnings.warn(f"PhysicalCBFFilter construction failed for {env_name}: {e}")

        for flt_name, flt in filters.items():
            # Use z for KCBF, x for Physical, either for Null
            inp_z = x if flt_name == "PhysicalCBFFilter" else z
            print(f"  Timing {flt_name:20s}  {env_name:14s}  u_dim={u_dim}")
            times = _time_filter(flt, inp_z, u, n_trials=n_trials, warmup=warmup)
            for ms in times:
                rows.append({
                    "env": env_name,
                    "filter": flt_name,
                    "u_dim": u_dim,
                    "z_dim": z_dim,
                    "trial_ms": ms,
                })

    return pd.DataFrame(rows)


def _summary_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["env", "filter", "u_dim"])["trial_ms"]
        .agg(mean="mean", std="std", p50="median",
             p95=lambda x: np.percentile(x, 95),
             p99=lambda x: np.percentile(x, 99))
        .reset_index()
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Benchmark per-iteration filter latency.")
    ap.add_argument("--out_dir", required=True, help="Output directory for CSV and plots")
    ap.add_argument("--n_trials", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running benchmark: {a.n_trials} trials, {a.warmup} warmup calls")
    df = run_full_benchmark(n_trials=a.n_trials, warmup=a.warmup)

    csv_path = out_dir / "filter_timing_raw.csv"
    df.to_csv(csv_path, index=False)
    print(f"Raw timings → {csv_path}")

    summary = _summary_table(df)
    summary_path = out_dir / "filter_timing_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary     → {summary_path}")
    print(summary.to_string(index=False))

    from robust_koopman_cbf_rl.plots.plot_filter_timing import plot_filter_timing
    plot_filter_timing(df, str(out_dir / "filter_timing.png"))
