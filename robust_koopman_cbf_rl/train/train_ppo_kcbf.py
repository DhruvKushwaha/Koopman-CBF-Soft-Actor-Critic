"""End-to-end KCBF-PPO training."""
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
from robust_koopman_cbf_rl.koopman.model import KoopmanModel
from robust_koopman_cbf_rl.cbf.robust_margin import RobustMargin
from robust_koopman_cbf_rl.cbf.qp_filter import KCBFQPFilter
from robust_koopman_cbf_rl.cbf.diagnostics import DiagnosticsBuffer
from robust_koopman_cbf_rl.cbf.factory import make_barrier
from robust_koopman_cbf_rl.agents.rollout_buffer import KCBFRolloutBuffer
from robust_koopman_cbf_rl.agents.ppo import KCBFPPOAgent
from robust_koopman_cbf_rl.train.train_koopman import build_env


def main(env_cfg_path, ppo_cfg_path, model_path, log_dir, total_steps=None,
         checkpoint_every: int = 50_000, seed=None):
    env_cfg = merge(EnvCfg, load_yaml(env_cfg_path))
    ppo_cfg = merge(PPOCfg, load_yaml(ppo_cfg_path))
    if total_steps is not None:
        ppo_cfg.total_steps = int(total_steps)
    if seed is not None:
        env_cfg.seed = int(seed)
    set_seed(env_cfg.seed)
    env = build_env(env_cfg)
    model = KoopmanModel.load(model_path)
    barrier = make_barrier(env_cfg.barrier)
    res_path = str(model_path).replace(".npz", "_residuals.npz")
    rm = RobustMargin(alpha=0.95, mode="global")
    if Path(res_path).exists():
        rm.update(np.load(res_path)["deltas"])
    else:
        warnings.warn(
            f"Residual file not found at {res_path}; using rho=0 (no robustness margin). "
            "Run train_koopman.py first to generate residuals.",
            UserWarning, stacklevel=2,
        )
        rm.update(np.zeros(10))
    flt = KCBFQPFilter(model, barrier, rm,
                       eta=ppo_cfg.eta, lam_slack=ppo_cfg.lam_slack,
                       u_min=env.action_space.low, u_max=env.action_space.high)
    diag_buf = DiagnosticsBuffer()
    dim_obs = env.observation_space.shape[0]
    dim_action = env.action_space.shape[0]
    agent = KCBFPPOAgent(dim_obs=dim_obs, dim_action=dim_action, dim_z=model.z_dim,
                         koopman_model=model, qp_filter=flt,
                         lam_h=ppo_cfg.lam_h, gamma=ppo_cfg.gamma, lam=ppo_cfg.lam,
                         lr=ppo_cfg.lr, actor_lr=ppo_cfg.actor_lr, critic_lr=ppo_cfg.critic_lr,
                         clip=ppo_cfg.clip, c_v=ppo_cfg.c_v, c_e=ppo_cfg.c_e)
    logger = CSVLogger(Path(log_dir) / "ppo_kcbf.csv")
    try:
        steps_done = 0; ep_ret = 0.0; ep_cost = 0.0
        obs, info = env.reset(seed=env_cfg.seed)
        y = info.get("raw_state", obs); z = model.lift(y[:model.observables.dim_y])
        while steps_done < ppo_cfg.total_steps:
            rb = KCBFRolloutBuffer(ppo_cfg.rollout_len, dim_obs, dim_action, model.z_dim)
            for _ in range(ppo_cfg.rollout_len):
                u_safe, u_nom, logp, v, diag = agent.select_action(obs)
                next_obs, reward, term, trunc, info = env.step(u_safe)
                y_next = info.get("raw_state", next_obs)
                z_next = model.lift(y_next[:model.observables.dim_y])
                rb.add(obs=obs, z=z, action_nom=u_nom, action_safe=u_safe,
                       logprob_nom=logp, value=v, reward=reward,
                       cost=info.get("cost", 0.0), done=bool(term or trunc),
                       h_value=diag["h_value"], cbf_gap=diag["cbf_gap"],
                       slack=diag["slack"], intervention=diag["intervention"])
                diag_buf.log(**diag)
                ep_ret += reward; ep_cost += info.get("cost", 0.0)
                obs, y, z = next_obs, y_next, z_next
                if term or trunc:
                    logger.log({"step": steps_done, "ep_return": ep_ret,
                                "ep_cost": ep_cost, **diag_buf.summary()})
                    diag_buf.reset()
                    obs, info = env.reset()
                    y = info.get("raw_state", obs); z = model.lift(y[:model.observables.dim_y])
                    ep_ret = 0.0; ep_cost = 0.0
                steps_done += 1
                if checkpoint_every > 0 and steps_done > 0 and steps_done % checkpoint_every == 0:
                    agent.save(Path(log_dir) / f"ppo_kcbf_step{steps_done}.pt")
                if steps_done >= ppo_cfg.total_steps:
                    break
            with torch.no_grad():
                _, _, _, v, _ = agent.select_action(obs)
            rb.compute_advantages(last_value=v, gamma=ppo_cfg.gamma, lam=ppo_cfg.lam)
            agent.update(rb.batched(), epochs=ppo_cfg.epochs, minibatch=ppo_cfg.minibatch)
    finally:
        agent.save(Path(log_dir) / "ppo_kcbf_final.pt")
        try:
            from robust_koopman_cbf_rl.train.evaluate import evaluate
            results = evaluate(env, agent, model, flt, n_episodes=10)
            pd.DataFrame(results).to_csv(Path(log_dir) / "eval_final.csv", index=False)
        except Exception as exc:
            warnings.warn(f"Post-training eval failed: {exc}")
        logger.close()
        env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_cfg", required=True)
    ap.add_argument("--ppo_cfg", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--total_steps", type=int, default=None)
    ap.add_argument("--checkpoint_every", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=None, help="Override seed from env_cfg")
    a = ap.parse_args()
    main(a.env_cfg, a.ppo_cfg, a.model, a.log_dir,
         total_steps=a.total_steps, checkpoint_every=a.checkpoint_every, seed=a.seed)
