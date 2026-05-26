"""CLI: load a trained agent, run evaluation episodes, save results CSV."""
from __future__ import annotations
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

from robust_koopman_cbf_rl.utils.config import load_yaml, merge, EnvCfg
from robust_koopman_cbf_rl.utils.seeding import set_seed
from robust_koopman_cbf_rl.koopman.model import KoopmanModel
from robust_koopman_cbf_rl.train.train_koopman import build_env


def _build_filter(model, env, res_path, barrier_cfg, eta: float, lam_slack: float):
    from robust_koopman_cbf_rl.cbf.robust_margin import RobustMargin
    from robust_koopman_cbf_rl.cbf.qp_filter import KCBFQPFilter
    from robust_koopman_cbf_rl.cbf.factory import make_barrier
    barrier = make_barrier(barrier_cfg)
    rm = RobustMargin(alpha=0.95, mode="global")
    if Path(res_path).exists():
        rm.update(np.load(res_path)["deltas"])
    else:
        warnings.warn(f"Residual file not found at {res_path}; using zero margin.")
        rm.update(np.zeros(10))
    return KCBFQPFilter(model, barrier, rm,
                        eta=eta, lam_slack=lam_slack,
                        u_min=env.action_space.low,
                        u_max=env.action_space.high)


def main(env_cfg_path, agent_type, agent_ckpt, model_path, log_dir, n_episodes=10,
         eta: float = 0.5, lam_slack: float = 1e3):
    env_cfg = merge(EnvCfg, load_yaml(env_cfg_path))
    set_seed(env_cfg.seed)
    env = build_env(env_cfg)
    model = KoopmanModel.load(model_path)
    res_path = str(model_path).replace(".npz", "_residuals.npz")
    flt = _build_filter(model, env, res_path, env_cfg.barrier, eta=eta, lam_slack=lam_slack)

    if agent_type == "sac":
        from robust_koopman_cbf_rl.agents.sac import KCBFSACAgent
        agent = KCBFSACAgent.load(agent_ckpt, koopman_model=model, qp_filter=flt)
    elif agent_type == "ppo":
        from robust_koopman_cbf_rl.agents.ppo import KCBFPPOAgent
        agent = KCBFPPOAgent.load(agent_ckpt, koopman_model=model, qp_filter=flt)
    else:
        raise ValueError(f"Unknown agent_type: {agent_type!r}")

    from robust_koopman_cbf_rl.train.evaluate import evaluate
    results = evaluate(env, agent, model, flt, n_episodes=n_episodes)
    out_path = Path(log_dir) / "eval_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Saved {len(results)} episodes → {out_path}")
    env.close()
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate a saved KCBF agent.")
    ap.add_argument("--env_cfg", required=True)
    ap.add_argument("--agent_type", choices=["sac", "ppo"], required=True)
    ap.add_argument("--agent_ckpt", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--n_episodes", type=int, default=10)
    ap.add_argument("--eta", type=float, default=0.5,
                    help="CBF decay rate (must match training value; default: 0.5)")
    ap.add_argument("--lam_slack", type=float, default=1e3,
                    help="QP slack penalty (must match training value; default: 1e3)")
    a = ap.parse_args()
    main(a.env_cfg, a.agent_type, a.agent_ckpt, a.model, a.log_dir, a.n_episodes,
         eta=a.eta, lam_slack=a.lam_slack)
