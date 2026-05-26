"""Roll out a trained policy, return per-episode metrics."""
from __future__ import annotations
import numpy as np

from robust_koopman_cbf_rl.cbf.diagnostics import DiagnosticsBuffer


def _fallback_metrics(rewards, costs, h_vals, viol, diag_summary):
    rewards = np.asarray(rewards, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    h_values = np.asarray(h_vals, dtype=np.float64)
    viol = np.asarray(viol, dtype=np.float64)
    out = {
        "return": float(rewards.sum()),
        "total_cost": float(costs.sum()),
        "violation_rate": float(viol.mean()) if len(viol) else 0.0,
        "episode_violation": float(viol.any()) if len(viol) else 0.0,
        "min_h_value": float(h_values.min()) if len(h_values) else float("nan"),
    }
    for k, v in (diag_summary or {}).items():
        out[k] = v
    return out


def _get_metrics_fn():
    try:
        from robust_koopman_cbf_rl.utils.metrics import compute_episode_metrics
        return compute_episode_metrics
    except ImportError:
        return _fallback_metrics


def evaluate(env, agent, model, qp_filter, n_episodes: int = 10):
    compute_metrics = _get_metrics_fn()
    results = []
    diag = DiagnosticsBuffer()
    for ep in range(n_episodes):
        obs, info = env.reset(seed=1000 + ep)
        y = info.get("raw_state", obs)
        z = model.lift(y[:model.observables.dim_y])
        rewards, costs, h_vals, viol = [], [], [], []
        done = False
        while not done:
            if hasattr(agent, "select_action"):
                result = agent.select_action(obs)
                u_safe, u_nom, d = result[0], result[1], result[-1]
            else:
                u_safe, d = qp_filter.project(z, np.zeros(env.action_space.shape))
                u_nom = None
            next_obs, r, term, trunc, info = env.step(u_safe)
            rewards.append(r); costs.append(info.get("cost", 0.0))
            h_vals.append(d["h_value"])
            viol.append(1.0 if d["h_value"] < 0 else 0.0)
            diag.log(**d)
            obs = next_obs
            y = info.get("raw_state", obs); z = model.lift(y[:model.observables.dim_y])
            done = bool(term) or bool(trunc)
        results.append(compute_metrics(rewards, costs, h_vals, viol, diag.summary()))
        diag.reset()
    return results
