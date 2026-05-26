"""SAC + reward-penalty baseline (no safety filter, cost subtracted from reward)."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import warnings

from robust_koopman_cbf_rl.utils.config import load_yaml, merge, SACCfg, EnvCfg
from robust_koopman_cbf_rl.utils.seeding import set_seed
from robust_koopman_cbf_rl.utils.logger import CSVLogger
from robust_koopman_cbf_rl.cbf.null_filter import NullFilter, NullModel
from robust_koopman_cbf_rl.agents.sac import KCBFSACAgent
from robust_koopman_cbf_rl.agents.replay_buffer import KCBFReplayBuffer
from robust_koopman_cbf_rl.train.evaluate_baseline import evaluate_baseline
from robust_koopman_cbf_rl.train.train_koopman import build_env


def main(env_cfg_path, sac_cfg_path, log_dir, lam_c=1.0, total_steps=None,
         checkpoint_every=50_000, seed=None):
    env_cfg = merge(EnvCfg, load_yaml(env_cfg_path))
    sac_cfg = merge(SACCfg, load_yaml(sac_cfg_path))
    if total_steps is not None:
        sac_cfg.total_steps = int(total_steps)
    lam_c = float(lam_c)
    if seed is not None:
        env_cfg.seed = int(seed)
    set_seed(env_cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = build_env(env_cfg)
    flt = NullFilter(u_min=env.action_space.low, u_max=env.action_space.high)
    model = NullModel()
    dim_obs = env.observation_space.shape[0]
    dim_action = env.action_space.shape[0]
    agent = KCBFSACAgent(dim_obs=dim_obs, dim_action=dim_action, dim_z=model.z_dim,
                         koopman_model=model, qp_filter=flt,
                         lam_h=0.0, gamma=sac_cfg.gamma, tau=sac_cfg.tau,
                         lr=sac_cfg.lr, actor_lr=sac_cfg.actor_lr, critic_lr=sac_cfg.critic_lr,
                         alpha=sac_cfg.alpha, device=device)
    buf = KCBFReplayBuffer(sac_cfg.buffer_capacity, dim_obs, dim_action, model.z_dim)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(Path(log_dir) / "sac_penalty.csv")
    rng = np.random.default_rng(env_cfg.seed)
    try:
        obs, info = env.reset(seed=env_cfg.seed)
        ep_ret = 0.0; ep_cost = 0.0; ep_penalized_ret = 0.0
        for step in range(sac_cfg.total_steps):
            if step < sac_cfg.warmup_steps:
                u_nom = env.action_space.sample()
                u_safe = np.clip(u_nom, env.action_space.low, env.action_space.high)
                diag = {"h_value": float("nan"), "cbf_gap": float("nan"),
                        "slack": 0.0, "intervention": False}
            else:
                u_safe, u_nom, diag = agent.select_action(obs)
            next_obs, reward, term, trunc, info = env.step(u_safe)
            cost = float(info.get("cost", 0.0))
            shaped_r = reward - lam_c * cost
            done = bool(term)
            buf.add(obs=obs, z=model.lift(obs[:model.observables.dim_y]),
                    action_nom=u_safe, action_safe=u_safe,
                    reward=shaped_r, cost=cost,
                    next_obs=next_obs, next_z=model.lift(next_obs[:model.observables.dim_y]),
                    done=done,
                    h_value=float("nan"), cbf_gap=float("nan"),
                    slack=0.0, intervention=False)
            ep_ret += reward; ep_cost += cost; ep_penalized_ret += shaped_r
            obs = next_obs
            if term or trunc:
                logger.log({"step": step, "ep_return": ep_ret, "ep_cost": ep_cost,
                            "ep_penalized_return": ep_penalized_ret})
                obs, info = env.reset()
                ep_ret = 0.0; ep_cost = 0.0; ep_penalized_ret = 0.0
            if step >= sac_cfg.warmup_steps and len(buf) >= sac_cfg.batch_size:
                agent.update(buf.sample(sac_cfg.batch_size, rng=rng))
            if checkpoint_every > 0 and step > 0 and step % checkpoint_every == 0:
                agent.save(Path(log_dir) / f"sac_penalty_step{step}.pt")
    finally:
        agent.save(Path(log_dir) / "sac_penalty_final.pt")
        try:
            results = evaluate_baseline(env, agent, n_episodes=10)
            pd.DataFrame(results).to_csv(Path(log_dir) / "eval_final.csv", index=False)
        except Exception as exc:
            warnings.warn(f"Post-training eval failed: {exc}")
        logger.close()
        env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_cfg", required=True)
    ap.add_argument("--sac_cfg", required=True)
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--lam_c", type=float, default=1.0)
    ap.add_argument("--total_steps", type=int, default=None)
    ap.add_argument("--checkpoint_every", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=None, help="Override seed from env_cfg")
    a = ap.parse_args()
    main(a.env_cfg, a.sac_cfg, a.log_dir, a.lam_c, a.total_steps, a.checkpoint_every,
         seed=a.seed)
