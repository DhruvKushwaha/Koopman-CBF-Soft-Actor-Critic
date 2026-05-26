"""End-to-end KCBF-SAC training."""
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
from robust_koopman_cbf_rl.koopman.model import KoopmanModel
from robust_koopman_cbf_rl.cbf.robust_margin import RobustMargin
from robust_koopman_cbf_rl.cbf.qp_filter import KCBFQPFilter
from robust_koopman_cbf_rl.cbf.diagnostics import DiagnosticsBuffer
from robust_koopman_cbf_rl.cbf.factory import make_barrier
from robust_koopman_cbf_rl.agents.replay_buffer import KCBFReplayBuffer
from robust_koopman_cbf_rl.agents.sac import KCBFSACAgent
from robust_koopman_cbf_rl.train.train_koopman import build_env


def main(env_cfg_path, sac_cfg_path, model_path, log_dir, total_steps=None,
         checkpoint_every: int = 50_000, seed=None):
    env_cfg = merge(EnvCfg, load_yaml(env_cfg_path))
    sac_cfg = merge(SACCfg, load_yaml(sac_cfg_path))
    if total_steps is not None:
        sac_cfg.total_steps = int(total_steps)
    if seed is not None:
        env_cfg.seed = int(seed)
    set_seed(env_cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = build_env(env_cfg)
    model = KoopmanModel.load(model_path)
    barrier = make_barrier(env_cfg.barrier)
    res_path = str(model_path).replace(".npz", "_residuals.npz")
    rm = RobustMargin(alpha=0.95, mode="global")
    if Path(res_path).exists():
        npz = np.load(res_path)
        rm.update(npz["deltas"])
    else:
        warnings.warn(
            f"Residual file not found at {res_path}; using rho=0 (no robustness margin). "
            "Run train_koopman.py first to generate residuals.",
            UserWarning, stacklevel=2,
        )
        rm.update(np.zeros(10))
    flt = KCBFQPFilter(model, barrier, rm,
                       eta=sac_cfg.eta, lam_slack=sac_cfg.lam_slack,
                       u_min=env.action_space.low, u_max=env.action_space.high)
    diag_buf = DiagnosticsBuffer()
    dim_obs = env.observation_space.shape[0]
    dim_action = env.action_space.shape[0]
    agent = KCBFSACAgent(dim_obs=dim_obs, dim_action=dim_action, dim_z=model.z_dim,
                         koopman_model=model, qp_filter=flt,
                         lam_h=sac_cfg.lam_h, gamma=sac_cfg.gamma, tau=sac_cfg.tau,
                         lr=sac_cfg.lr, actor_lr=sac_cfg.actor_lr, critic_lr=sac_cfg.critic_lr,
                         alpha=sac_cfg.alpha, device=device)
    buf = KCBFReplayBuffer(sac_cfg.buffer_capacity, dim_obs, dim_action, model.z_dim)
    logger = CSVLogger(Path(log_dir) / "sac_kcbf.csv")
    rng = np.random.default_rng(env_cfg.seed)

    try:
        obs, info = env.reset(seed=env_cfg.seed)
        y = info.get("raw_state", obs); z = model.lift(y[:model.observables.dim_y])
        ep_ret = 0.0; ep_cost = 0.0
        for step in range(sac_cfg.total_steps):
            if step < sac_cfg.warmup_steps:
                u_nom = env.action_space.sample()
                u_safe, diag = flt.project(z, u_nom)
            else:
                u_safe, u_nom, diag = agent.select_action(obs)
            next_obs, reward, term, trunc, info = env.step(u_safe)
            y_next = info.get("raw_state", next_obs)
            z_next = model.lift(y_next[:model.observables.dim_y])
            done = bool(term)
            buf.add(obs=obs, z=z, action_nom=u_nom, action_safe=u_safe,
                    reward=reward, cost=info.get("cost", 0.0),
                    next_obs=next_obs, next_z=z_next, done=done,
                    h_value=diag["h_value"], cbf_gap=diag["cbf_gap"],
                    slack=diag["slack"], intervention=diag["intervention"])
            diag_buf.log(**diag)
            ep_ret += reward; ep_cost += info.get("cost", 0.0)
            obs, y, z = next_obs, y_next, z_next
            if term or trunc:
                logger.log({"step": step, "ep_return": ep_ret, "ep_cost": ep_cost,
                            **diag_buf.summary()})
                diag_buf.reset()
                obs, info = env.reset()
                y = info.get("raw_state", obs); z = model.lift(y[:model.observables.dim_y])
                ep_ret = 0.0; ep_cost = 0.0
            if step >= sac_cfg.warmup_steps and len(buf) >= sac_cfg.batch_size:
                batch = buf.sample(sac_cfg.batch_size, rng=rng)
                agent.update(batch)
            if checkpoint_every > 0 and step > 0 and step % checkpoint_every == 0:
                agent.save(Path(log_dir) / f"sac_kcbf_step{step}.pt")
    finally:
        agent.save(Path(log_dir) / "sac_kcbf_final.pt")
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
    ap.add_argument("--sac_cfg", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--total_steps", type=int, default=None)
    ap.add_argument("--checkpoint_every", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=None, help="Override seed from env_cfg")
    a = ap.parse_args()
    main(a.env_cfg, a.sac_cfg, a.model, a.log_dir,
         total_steps=a.total_steps, checkpoint_every=a.checkpoint_every, seed=a.seed)
