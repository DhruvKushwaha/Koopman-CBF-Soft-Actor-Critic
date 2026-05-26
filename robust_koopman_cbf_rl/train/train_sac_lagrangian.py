"""SAC + Lagrangian dual baseline (no safety filter, adaptive cost penalty)."""
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
from robust_koopman_cbf_rl.baselines.lagrangian_rl import LagrangianDual
from robust_koopman_cbf_rl.train.evaluate_baseline import evaluate_baseline
from robust_koopman_cbf_rl.train.train_koopman import build_env


def main(env_cfg_path, sac_cfg_path, log_dir, dual_lr=0.01, cost_budget=0.0,
         total_steps=None, checkpoint_every=50_000, seed=None):
    env_cfg = merge(EnvCfg, load_yaml(env_cfg_path))
    sac_cfg = merge(SACCfg, load_yaml(sac_cfg_path))
    if total_steps is not None:
        sac_cfg.total_steps = int(total_steps)
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
    dual = LagrangianDual(init_value=0.0, lr=float(dual_lr), budget=float(cost_budget))
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(Path(log_dir) / "sac_lagrangian.csv")
    rng = np.random.default_rng(env_cfg.seed)
    try:
        obs, info = env.reset(seed=env_cfg.seed)
        ep_ret = 0.0; ep_cost = 0.0; ep_steps = 0
        # Accumulate episode transitions before pushing to replay buffer
        ep_transitions = []
        for step in range(sac_cfg.total_steps):
            if step < sac_cfg.warmup_steps:
                u_nom = env.action_space.sample()
                u_safe = np.clip(u_nom, env.action_space.low, env.action_space.high)
            else:
                u_safe, u_nom, diag = agent.select_action(obs)
            next_obs, reward, term, trunc, info = env.step(u_safe)
            cost = float(info.get("cost", 0.0))
            done = bool(term)
            ep_transitions.append({
                "obs": obs.copy(), "next_obs": next_obs.copy(),
                "action": u_safe.copy(), "reward": reward, "cost": cost, "done": done,
            })
            ep_ret += reward; ep_cost += cost; ep_steps += 1
            obs = next_obs
            if term or trunc:
                # Update dual variable after episode, then shape all stored transitions
                mean_cost = ep_cost / max(ep_steps, 1)
                dual.update(mean_cost=mean_cost)
                lam_c = dual.value
                for tr in ep_transitions:
                    shaped_r = tr["reward"] - lam_c * tr["cost"]
                    buf.add(
                        obs=tr["obs"],
                        z=model.lift(tr["obs"][:model.observables.dim_y]),
                        action_nom=tr["action"], action_safe=tr["action"],
                        reward=shaped_r, cost=tr["cost"],
                        next_obs=tr["next_obs"],
                        next_z=model.lift(tr["next_obs"][:model.observables.dim_y]),
                        done=tr["done"],
                        h_value=float("nan"), cbf_gap=float("nan"),
                        slack=0.0, intervention=False,
                    )
                logger.log({"step": step, "ep_return": ep_ret, "ep_cost": ep_cost,
                            "dual_value": dual.value})
                obs, info = env.reset()
                ep_ret = 0.0; ep_cost = 0.0; ep_steps = 0
                ep_transitions = []
            if step >= sac_cfg.warmup_steps and len(buf) >= sac_cfg.batch_size:
                agent.update(buf.sample(sac_cfg.batch_size, rng=rng))
            if checkpoint_every > 0 and step > 0 and step % checkpoint_every == 0:
                agent.save(Path(log_dir) / f"sac_lagrangian_step{step}.pt")
    finally:
        agent.save(Path(log_dir) / "sac_lagrangian_final.pt")
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
    ap.add_argument("--dual_lr", type=float, default=0.01)
    ap.add_argument("--cost_budget", type=float, default=0.0)
    ap.add_argument("--total_steps", type=int, default=None)
    ap.add_argument("--checkpoint_every", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=None, help="Override seed from env_cfg")
    a = ap.parse_args()
    main(a.env_cfg, a.sac_cfg, a.log_dir, a.dual_lr, a.cost_budget,
         a.total_steps, a.checkpoint_every, seed=a.seed)
