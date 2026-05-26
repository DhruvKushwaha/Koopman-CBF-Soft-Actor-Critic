"""Unconstrained PPO baseline (no safety filter)."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import warnings

from robust_koopman_cbf_rl.utils.config import load_yaml, merge, PPOCfg, EnvCfg
from robust_koopman_cbf_rl.utils.seeding import set_seed
from robust_koopman_cbf_rl.utils.logger import CSVLogger
from robust_koopman_cbf_rl.cbf.null_filter import NullFilter, NullModel
from robust_koopman_cbf_rl.agents.ppo import KCBFPPOAgent
from robust_koopman_cbf_rl.agents.rollout_buffer import KCBFRolloutBuffer
from robust_koopman_cbf_rl.train.evaluate_baseline import evaluate_baseline
from robust_koopman_cbf_rl.train.train_koopman import build_env


def main(env_cfg_path, ppo_cfg_path, log_dir, total_steps=None, checkpoint_every=50_000,
         seed=None):
    env_cfg = merge(EnvCfg, load_yaml(env_cfg_path))
    ppo_cfg = merge(PPOCfg, load_yaml(ppo_cfg_path))
    if total_steps is not None:
        ppo_cfg.total_steps = int(total_steps)
    if seed is not None:
        env_cfg.seed = int(seed)
    set_seed(env_cfg.seed)
    env = build_env(env_cfg)
    flt = NullFilter(u_min=env.action_space.low, u_max=env.action_space.high)
    model = NullModel()
    dim_obs = env.observation_space.shape[0]
    dim_action = env.action_space.shape[0]
    agent = KCBFPPOAgent(dim_obs=dim_obs, dim_action=dim_action, dim_z=model.z_dim,
                         koopman_model=model, qp_filter=flt,
                         lam_h=0.0, gamma=ppo_cfg.gamma, lam=ppo_cfg.lam,
                         lr=ppo_cfg.lr, actor_lr=ppo_cfg.actor_lr, critic_lr=ppo_cfg.critic_lr,
                         clip=ppo_cfg.clip, c_v=ppo_cfg.c_v, c_e=ppo_cfg.c_e)
    logger = CSVLogger(Path(log_dir) / "ppo_baseline.csv")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    try:
        steps_done = 0; ep_ret = 0.0; ep_cost = 0.0
        obs, info = env.reset(seed=env_cfg.seed)
        while steps_done < ppo_cfg.total_steps:
            rb = KCBFRolloutBuffer(ppo_cfg.rollout_len, dim_obs, dim_action, model.z_dim)
            for _ in range(ppo_cfg.rollout_len):
                u_safe, u_nom, logp, v, diag = agent.select_action(obs)
                next_obs, reward, term, trunc, info = env.step(u_safe)
                cost = float(info.get("cost", 0.0))
                rb.add(obs=obs, z=model.lift(obs[:model.observables.dim_y]),
                       action_nom=u_nom, action_safe=u_safe,
                       logprob_nom=logp, value=v, reward=reward,
                       cost=cost, done=bool(term or trunc),
                       h_value=float("nan"), cbf_gap=float("nan"),
                       slack=0.0, intervention=False)
                ep_ret += reward; ep_cost += cost
                obs = next_obs
                if term or trunc:
                    logger.log({"step": steps_done, "ep_return": ep_ret, "ep_cost": ep_cost})
                    obs, info = env.reset()
                    ep_ret = 0.0; ep_cost = 0.0
                steps_done += 1
                if checkpoint_every > 0 and steps_done > 0 and steps_done % checkpoint_every == 0:
                    agent.save(Path(log_dir) / f"ppo_baseline_step{steps_done}.pt")
                if steps_done >= ppo_cfg.total_steps:
                    break
            with torch.no_grad():
                _, _, _, v, _ = agent.select_action(obs)
            rb.compute_advantages(last_value=v, gamma=ppo_cfg.gamma, lam=ppo_cfg.lam)
            agent.update(rb.batched(), epochs=ppo_cfg.epochs, minibatch=ppo_cfg.minibatch)
    finally:
        agent.save(Path(log_dir) / "ppo_baseline_final.pt")
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
    ap.add_argument("--ppo_cfg", required=True)
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--total_steps", type=int, default=None)
    ap.add_argument("--checkpoint_every", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=None, help="Override seed from env_cfg")
    a = ap.parse_args()
    main(a.env_cfg, a.ppo_cfg, a.log_dir, a.total_steps, a.checkpoint_every, seed=a.seed)
