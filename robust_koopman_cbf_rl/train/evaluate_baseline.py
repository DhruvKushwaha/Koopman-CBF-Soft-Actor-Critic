"""Evaluate a baseline agent (classical controller or unconstrained RL)."""
from __future__ import annotations
import inspect
import numpy as np

_ACCEPTS_INFO: dict[type, bool] = {}


def _agent_accepts_info(agent) -> bool:
    cls = type(agent)
    if cls not in _ACCEPTS_INFO:
        sig = inspect.signature(agent.select_action)
        _ACCEPTS_INFO[cls] = "info" in sig.parameters
    return _ACCEPTS_INFO[cls]


def evaluate_baseline(env, agent, n_episodes: int = 10, env_cfg=None):
    """Run n_episodes and return per-episode metric dicts.

    Accepts agents whose select_action returns:
      - np.ndarray (LQR, PID), or
      - tuple (u_safe, ...) (RL agents).
    """
    accepts_info = _agent_accepts_info(agent)
    results = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=1000 + ep)
        rewards, costs, viols = [], [], []
        done = False
        while not done:
            result = agent.select_action(obs, info=info) if accepts_info else agent.select_action(obs)
            action = result[0] if isinstance(result, tuple) else result
            action = np.asarray(action, dtype=np.float64)
            if hasattr(env, "action_space"):
                action = np.clip(action, env.action_space.low, env.action_space.high)
            next_obs, r, term, trunc, info = env.step(action)
            cost = float(info.get("cost", 0.0))
            rewards.append(float(r))
            costs.append(cost)
            viols.append(1.0 if cost > 0 else 0.0)
            obs = next_obs
            done = bool(term) or bool(trunc)
        results.append({
            "return": float(np.sum(rewards)),
            "total_cost": float(np.sum(costs)),
            "violation_rate": float(np.mean(viols)) if viols else 0.0,
            "episode_violation": float(any(v > 0 for v in viols)),
            "min_h_value": float("nan"),
            "intervention_rate": float("nan"),
        })
    return results
