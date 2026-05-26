"""Collect rollouts, fit EDMD, compute residuals + margin, save model."""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np

from robust_koopman_cbf_rl.utils.config import load_yaml, merge, KoopmanCfg, EnvCfg
from robust_koopman_cbf_rl.utils.seeding import set_seed
from robust_koopman_cbf_rl.koopman.dataset import KoopmanDataset
from robust_koopman_cbf_rl.koopman.observables import RBFObservables
from robust_koopman_cbf_rl.koopman.fit_edmd import fit_edmd
from robust_koopman_cbf_rl.koopman.model import KoopmanModel
from robust_koopman_cbf_rl.koopman.residuals import compute_residuals, compute_robust_margin
from robust_koopman_cbf_rl.train.collect_koopman_data import collect_rollouts
from robust_koopman_cbf_rl.cbf.factory import make_barrier


def build_env(env_cfg: EnvCfg):
    if env_cfg.kind == "safe_control_gym":
        from robust_koopman_cbf_rl.envs.safe_control_gym_wrapper import make_safe_control_gym_env
        return make_safe_control_gym_env(env_cfg.env_id, env_cfg.task_config, seed=env_cfg.seed)
    else:
        from robust_koopman_cbf_rl.envs.safety_gymnasium_wrapper import make_safety_gym_env
        return make_safety_gym_env(env_cfg.env_id, velocity_limit=env_cfg.velocity_limit, seed=env_cfg.seed)


def _auto_bandwidth(Y: np.ndarray, n_subsample: int = 2000) -> float:
    """Median pairwise distance heuristic; bandwidth ≤ 0 in config triggers this."""
    rng = np.random.default_rng(42)
    idx = rng.choice(len(Y), size=min(n_subsample, len(Y)), replace=False)
    sub = Y[idx]
    diff = sub[:, None, :] - sub[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    upper = dists[np.triu_indices(len(sub), k=1)]
    return float(max(np.median(upper) / np.sqrt(2.0), 1e-3))


def main(env_cfg_path: str, koopman_cfg_path: str, out_path: str):
    env_cfg = merge(EnvCfg, load_yaml(env_cfg_path))
    k_cfg = merge(KoopmanCfg, load_yaml(koopman_cfg_path))
    set_seed(k_cfg.seed)
    env = build_env(env_cfg)
    barrier = make_barrier(env_cfg.barrier)
    extra_quadratic = list(barrier.extra_features().get("quadratic_indices", []))
    ds = collect_rollouts(env, num_steps=k_cfg.collection_steps, seed=k_cfg.seed)
    Y, U, Yp = ds.as_arrays()
    bandwidth = _auto_bandwidth(Y) if k_cfg.bandwidth <= 0 else k_cfg.bandwidth
    print(f"  bandwidth={'auto→' if k_cfg.bandwidth <= 0 else ''}{bandwidth:.4f}  "
          f"n_rbf={k_cfg.n_rbf}  collection_steps={k_cfg.collection_steps}  ridge={k_cfg.ridge}")
    obs = RBFObservables(dim_y=ds.dim_y, n_rbf=k_cfg.n_rbf,
                         bandwidth=bandwidth,
                         extra_quadratic_indices=extra_quadratic, seed=k_cfg.seed)
    obs.fit_centers(Y)
    Z = obs.lift_batch(Y); Zp = obs.lift_batch(Yp)
    A, B = fit_edmd(Z, Zp, U, reg=k_cfg.ridge)
    model = KoopmanModel(obs, A, B)
    # Use real c from barrier so projection direction matches deployment.
    c_real = np.asarray(
        barrier.lifted_barrier_coeffs(z_dim=obs.z_dim, dim_y=ds.dim_y, n_rbf=k_cfg.n_rbf)[0],
        dtype=np.float64,
    )
    deltas = compute_residuals(model, Y, U, Yp, c=c_real)
    rho = compute_robust_margin(deltas, alpha=k_cfg.alpha)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    np.savez(str(out_path).replace(".npz", "_residuals.npz"),
             deltas=deltas, rho=rho, alpha=k_cfg.alpha)
    env.close()
    return model, rho


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_cfg", required=True)
    ap.add_argument("--koopman_cfg", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.env_cfg, a.koopman_cfg, a.out)
