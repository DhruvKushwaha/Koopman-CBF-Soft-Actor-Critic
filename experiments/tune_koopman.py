"""Koopman hyperparameter sweep for all supported environments.

Tests the effect of n_rbf, collection_steps, bandwidth, and ridge on
one-step prediction accuracy and barrier residual rho (the key CBF safety margin).

Metrics reported per config:
  r2_z       — R² on held-out lifted state (overall fit quality)
  rho_95     — 95th percentile of projected barrier residuals (drives CBF safety margin)
  rho_max    — max projected residual (worst-case safety margin)
  res_mean   — mean Euclidean residual ||z_next - A z + B u||
  h_mae      — mean absolute error on barrier value prediction
  rollout_5  — 5-step ahead barrier prediction error (multi-step stability)

Usage:
    # Safety Gym (high-dim, large grid)
    python experiments/tune_koopman.py --envs safety_halfcheetah safety_walker
    python experiments/tune_koopman.py --envs safety_halfcheetah --n_rbf 64 128 256 \
        --steps 50000 200000 --ridge 1e-6 1e-4

    # Quadrotor 2D (low-dim: use smaller n_rbf and fewer collection steps)
    python experiments/tune_koopman.py \
        --envs quadrotor2d_stab quadrotor2d_track \
        --n_rbf 16 32 64 128 \
        --steps 5000 10000 20000 50000 \
        --ridge 1e-6 1e-4 1e-2

    # Both env families together
    python experiments/tune_koopman.py \
        --envs quadrotor2d_stab quadrotor2d_track safety_walker \
        --n_rbf 32 64 128 \
        --steps 10000 50000 200000 \
        --ridge 1e-6 1e-4
"""
from __future__ import annotations
import argparse
import sys
import time
import warnings
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from robust_koopman_cbf_rl.utils.config import load_yaml, merge, EnvCfg, KoopmanCfg
from robust_koopman_cbf_rl.koopman.dataset import KoopmanDataset
from robust_koopman_cbf_rl.koopman.observables import RBFObservables
from robust_koopman_cbf_rl.koopman.fit_edmd import fit_edmd
from robust_koopman_cbf_rl.koopman.model import KoopmanModel
from robust_koopman_cbf_rl.koopman.residuals import compute_residuals
from robust_koopman_cbf_rl.cbf.factory import make_barrier
from robust_koopman_cbf_rl.train.train_koopman import build_env
from robust_koopman_cbf_rl.train.collect_koopman_data import collect_rollouts

_CFG = _ROOT / "robust_koopman_cbf_rl" / "configs"

_ENV_CFG = {
    "quadrotor2d_stab":   str(_CFG / "env_quadrotor2d_stab.yaml"),
    "quadrotor2d_track":  str(_CFG / "env_quadrotor2d_track.yaml"),
    "safety_halfcheetah": str(_CFG / "env_safety_halfcheetah.yaml"),
    "safety_walker":      str(_CFG / "env_safety_walker.yaml"),
}

# Recommended sweep grids per env family.
# Quadrotor 2D: state_dim=6 — smaller n_rbf avoids overparameterising the 6D EDMD problem;
#               near-hover dynamics are nearly linear so 5k–20k steps suffice.
# Safety Gym: state_dim=17 — needs larger dictionaries and more diverse data.
_ENV_DEFAULTS: dict[str, dict] = {
    "quadrotor2d_stab":  {"n_rbf": [16, 32, 64, 128],
                           "steps": [5_000, 10_000, 20_000, 50_000],
                           "ridge": [1e-6, 1e-4, 1e-2]},
    "quadrotor2d_track": {"n_rbf": [16, 32, 64, 128],
                           "steps": [5_000, 10_000, 20_000, 50_000],
                           "ridge": [1e-6, 1e-4, 1e-2]},
    "safety_halfcheetah":{"n_rbf": [64, 128, 256, 512],
                           "steps": [50_000, 100_000, 200_000, 500_000],
                           "ridge": [1e-6, 1e-4, 1e-2]},
    "safety_walker":     {"n_rbf": [64, 128, 256, 512],
                           "steps": [50_000, 100_000, 200_000, 500_000],
                           "ridge": [1e-6, 1e-4, 1e-2]},
}

_KOOPMAN_CFG = str(_CFG / "koopman.yaml")

# ── Data collection ────────────────────────────────────────────────────────────

def _data_cache_path(env_name: str, n_steps: int) -> Path:
    return _ROOT / "logs" / "koopman_tuning" / env_name / f"data_{n_steps}.npz"


def get_data(env_name: str, n_steps: int, seed: int = 0) -> KoopmanDataset:
    """Load cached data or collect and cache it."""
    cache = _data_cache_path(env_name, n_steps)
    if cache.exists():
        print(f"  [cache] {env_name} data n={n_steps}")
        return KoopmanDataset.load(cache)

    print(f"  [collect] {env_name} {n_steps} steps ...", flush=True)
    env_cfg = merge(EnvCfg, load_yaml(_ENV_CFG[env_name]))
    env = build_env(env_cfg)
    t0 = time.time()
    ds = collect_rollouts(env, num_steps=n_steps, seed=seed)
    env.close()
    print(f"    collected {len(ds)} transitions in {time.time()-t0:.1f}s")
    cache.parent.mkdir(parents=True, exist_ok=True)
    ds.save(cache)
    return ds


# ── Bandwidth heuristic ────────────────────────────────────────────────────────

def _auto_bandwidth(Y: np.ndarray, n_subsample: int = 2000) -> float:
    """Median pairwise distance heuristic (standard for RBF kernels)."""
    rng = np.random.default_rng(42)
    idx = rng.choice(len(Y), size=min(n_subsample, len(Y)), replace=False)
    sub = Y[idx]
    # Vectorized pairwise distances without sklearn dependency
    diff = sub[:, None, :] - sub[None, :, :]          # (n, n, d)
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))        # (n, n)
    upper = dists[np.triu_indices(len(sub), k=1)]
    median = float(np.median(upper))
    return max(median / np.sqrt(2.0), 1e-3)            # Silverman rule-of-thumb


# ── Model fitting and evaluation ───────────────────────────────────────────────

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _multistep_barrier_error(model: KoopmanModel, Y_seq: np.ndarray,
                              U_seq: np.ndarray, c: np.ndarray,
                              k: int = 5) -> float:
    """Mean |c^T z_k_pred - c^T z_k_true| over all valid sequences of length k."""
    N = len(Y_seq) - k
    if N <= 0:
        return float("nan")
    errors = []
    Z_true = model.lift_batch(Y_seq)
    for i in range(0, N, k):                      # stride by k to get independent seqs
        z = Z_true[i].copy()
        for step in range(k):
            z = model.A @ z + model.B @ U_seq[i + step]
        h_pred = float(c @ z)
        h_true = float(c @ Z_true[i + k])
        errors.append(abs(h_pred - h_true))
    return float(np.mean(errors)) if errors else float("nan")


def fit_and_eval(
    Y_train: np.ndarray, U_train: np.ndarray, Yp_train: np.ndarray,
    Y_test: np.ndarray,  U_test: np.ndarray,  Yp_test: np.ndarray,
    n_rbf: int, bandwidth: float, ridge: float,
    c: np.ndarray, extra_quad: list[int], seed: int = 0,
) -> dict:
    dim_y = Y_train.shape[1]
    t0 = time.time()

    obs = RBFObservables(dim_y=dim_y, n_rbf=n_rbf, bandwidth=bandwidth,
                         extra_quadratic_indices=extra_quad, seed=seed)
    obs.fit_centers(Y_train)

    Z_tr  = obs.lift_batch(Y_train)
    Zp_tr = obs.lift_batch(Yp_train)
    Z_te  = obs.lift_batch(Y_test)
    Zp_te = obs.lift_batch(Yp_test)

    A, B = fit_edmd(Z_tr, Zp_tr, U_train, reg=ridge)
    model = KoopmanModel(obs, A, B)

    # One-step predictions on test set
    Zp_pred = Z_te @ A.T + U_test @ B.T
    r2 = _r2(Zp_te, Zp_pred)

    # Euclidean residuals
    R = Zp_te - Zp_pred
    res_norms = np.linalg.norm(R, axis=1)

    # Projected barrier residuals (what actually enters rho in KCBF)
    proj = np.abs(R @ c)
    rho_95  = float(np.percentile(proj, 95))
    rho_max = float(np.max(proj))

    # Barrier value MAE on test set
    h_pred_te = Zp_pred @ c
    h_true_te = Zp_te   @ c
    h_mae = float(np.mean(np.abs(h_pred_te - h_true_te)))

    # 5-step ahead barrier prediction error
    rollout_5 = _multistep_barrier_error(model, Y_test, U_test, c, k=5)

    return {
        "n_rbf":      n_rbf,
        "z_dim":      obs.z_dim,
        "bandwidth":  round(bandwidth, 4),
        "ridge":      ridge,
        "r2_z":       round(r2, 4),
        "res_mean":   round(float(np.mean(res_norms)), 6),
        "rho_95":     round(rho_95, 6),
        "rho_max":    round(rho_max, 6),
        "h_mae":      round(h_mae, 6),
        "rollout_5":  round(rollout_5, 6),
        "fit_sec":    round(time.time() - t0, 2),
    }


# ── Per-env sweep ──────────────────────────────────────────────────────────────

def sweep_env(
    env_name: str,
    collection_steps_list: list[int],
    n_rbf_list: list[int],
    ridge_list: list[float],
    test_frac: float = 0.2,
    seed: int = 0,
) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"ENV: {env_name}")
    print(f"{'='*60}")

    env_cfg = merge(EnvCfg, load_yaml(_ENV_CFG[env_name]))
    barrier = make_barrier(env_cfg.barrier)
    extra_quad: list[int] = list(barrier.extra_features().get("quadratic_indices", []))

    rows = []

    for n_steps in sorted(collection_steps_list):
        ds = get_data(env_name, n_steps, seed=seed)
        Y, U, Yp = ds.as_arrays()

        # Auto bandwidth from this data
        bw_auto = _auto_bandwidth(Y)
        print(f"\n  n_steps={n_steps}  auto_bandwidth={bw_auto:.4f}  "
              f"dim_y={Y.shape[1]}  dim_u={U.shape[1]}")

        # Train/test split (sequential — no shuffle to preserve temporal structure)
        n_test = max(1, int(len(Y) * test_frac))
        Y_tr, U_tr, Yp_tr = Y[:-n_test], U[:-n_test], Yp[:-n_test]
        Y_te, U_te, Yp_te = Y[-n_test:], U[-n_test:], Yp[-n_test:]

        for n_rbf, ridge in product(n_rbf_list, ridge_list):
            # Get barrier coefficients for this specific n_rbf
            obs_tmp = RBFObservables(dim_y=Y.shape[1], n_rbf=n_rbf,
                                     extra_quadratic_indices=extra_quad, seed=seed)
            obs_tmp.fit_centers(Y_tr)
            z_dim_tmp = obs_tmp.z_dim
            c = np.asarray(
                barrier.lifted_barrier_coeffs(
                    z_dim=z_dim_tmp, dim_y=Y.shape[1], n_rbf=n_rbf
                )[0],
                dtype=np.float64,
            )

            metrics = fit_and_eval(
                Y_tr, U_tr, Yp_tr, Y_te, U_te, Yp_te,
                n_rbf=n_rbf, bandwidth=bw_auto, ridge=ridge,
                c=c, extra_quad=extra_quad, seed=seed,
            )
            metrics["env"]     = env_name
            metrics["n_steps"] = n_steps
            rows.append(metrics)
            print(f"    n_rbf={n_rbf:4d}  ridge={ridge:.0e}  "
                  f"r2={metrics['r2_z']:.4f}  rho_95={metrics['rho_95']:.4f}  "
                  f"h_mae={metrics['h_mae']:.4f}  rollout5={metrics['rollout_5']:.4f}  "
                  f"({metrics['fit_sec']:.1f}s)")

    return pd.DataFrame(rows)


# ── Results display ────────────────────────────────────────────────────────────

def _print_top(df: pd.DataFrame, env_name: str, n: int = 5) -> None:
    sub = df[df["env"] == env_name].copy()
    sub = sub.sort_values("rho_95").reset_index(drop=True)
    print(f"\n{'─'*80}")
    print(f"TOP {n} configs for {env_name}  (sorted by rho_95 ↓)")
    print(f"{'─'*80}")
    cols = ["n_steps", "n_rbf", "bandwidth", "ridge",
            "r2_z", "rho_95", "rho_max", "h_mae", "rollout_5"]
    print(sub[cols].head(n).to_string(index=False))
    best = sub.iloc[0]
    print(f"\n★ Recommended config for {env_name}:")
    print(f"  collection_steps: {int(best['n_steps'])}")
    print(f"  n_rbf:            {int(best['n_rbf'])}")
    print(f"  bandwidth:        {best['bandwidth']}  (auto from data)")
    print(f"  ridge:            {best['ridge']}")
    print(f"  → rho_95={best['rho_95']:.4f}  r2={best['r2_z']:.4f}  "
          f"rollout5={best['rollout_5']:.4f}")


def _compare_vs_baseline(df: pd.DataFrame) -> None:
    """Print improvement factor vs current baseline (n_rbf=64, n_steps=50k)."""
    print(f"\n{'═'*80}")
    print("IMPROVEMENT vs CURRENT BASELINE (n_rbf=64, n_steps=50k)")
    print(f"{'═'*80}")
    for env_name in df["env"].unique():
        sub = df[df["env"] == env_name]
        baseline = sub[(sub["n_rbf"] == 64) & (sub["n_steps"] == 50_000)]
        if baseline.empty:
            continue
        base_rho = baseline["rho_95"].values[0]
        best = sub.sort_values("rho_95").iloc[0]
        improvement = base_rho / max(best["rho_95"], 1e-9)
        print(f"  {env_name}: baseline rho_95={base_rho:.4f} → "
              f"best rho_95={best['rho_95']:.4f}  ({improvement:.1f}× reduction)  "
              f"[n_rbf={int(best['n_rbf'])}, n_steps={int(best['n_steps'])}, "
              f"ridge={best['ridge']:.0e}]")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--envs", nargs="+",
                    default=["safety_halfcheetah", "safety_walker"],
                    choices=list(_ENV_CFG))
    ap.add_argument("--n_rbf",  type=int, nargs="+", default=None,
                    help="RBF count grid. Default: per-env from _ENV_DEFAULTS.")
    ap.add_argument("--steps",  type=int, nargs="+", default=None,
                    help="Collection steps grid. Default: per-env from _ENV_DEFAULTS.")
    ap.add_argument("--ridge",  type=float, nargs="+", default=None,
                    help="Ridge regularisation grid. Default: per-env from _ENV_DEFAULTS.")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--out",    default="logs/koopman_tuning/results.csv")
    a = ap.parse_args()

    all_dfs = []
    for env_name in a.envs:
        defaults = _ENV_DEFAULTS[env_name]
        n_rbf_list = a.n_rbf  or defaults["n_rbf"]
        steps_list = a.steps  or defaults["steps"]
        ridge_list = a.ridge  or defaults["ridge"]

        n_configs = len(steps_list) * len(n_rbf_list) * len(ridge_list)
        print(f"\nKoopman tuning: {env_name}  ({n_configs} configs)")
        print(f"  n_rbf:  {n_rbf_list}")
        print(f"  steps:  {steps_list}")
        print(f"  ridge:  {ridge_list}")

        df = sweep_env(
            env_name,
            collection_steps_list=steps_list,
            n_rbf_list=n_rbf_list,
            ridge_list=ridge_list,
            test_frac=a.test_frac,
            seed=a.seed,
        )
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"\nFull results → {out_path}")

    for env_name in a.envs:
        _print_top(combined, env_name, n=5)

    _compare_vs_baseline(combined)


if __name__ == "__main__":
    main()
