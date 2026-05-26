"""Collect rollouts from an env with a random policy and save dataset."""
from __future__ import annotations
from pathlib import Path
import numpy as np

from robust_koopman_cbf_rl.koopman.dataset import KoopmanDataset


def collect_rollouts(env, num_steps: int, policy=None, seed: int = 0) -> KoopmanDataset:
    rng = np.random.default_rng(seed)
    dim_u = env.action_space.shape[0]
    obs, info = env.reset(seed=seed)
    y = np.asarray(info.get("raw_state", obs), dtype=np.float64)
    # Use raw_state dim, not observation_space dim: tracking envs augment obs
    # with goal/reference (e.g. cartpole_track obs=8 but physical state=4).
    dim_y = y.shape[0]
    ds = KoopmanDataset(dim_y=dim_y, dim_u=dim_u)
    steps = 0
    while steps < num_steps:
        if policy is None:
            lo, hi = env.action_space.low, env.action_space.high
            a = rng.uniform(lo, hi)
        else:
            a = policy(obs)
        a = np.asarray(a, dtype=np.float64)
        next_obs, _, term, trunc, info = env.step(a)
        y_next = np.asarray(info.get("raw_state", next_obs), dtype=np.float64)
        ds.add(y, a, y_next)
        obs, y = next_obs, y_next
        steps += 1
        if term or trunc:
            obs, info = env.reset()
            y = np.asarray(info.get("raw_state", obs), dtype=np.float64)
    return ds


def main(env_factory, out_path: str, num_steps: int = 50_000, seed: int = 0):
    env = env_factory()
    ds = collect_rollouts(env, num_steps=num_steps, seed=seed)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ds.save(out_path)
    env.close()
    return ds
