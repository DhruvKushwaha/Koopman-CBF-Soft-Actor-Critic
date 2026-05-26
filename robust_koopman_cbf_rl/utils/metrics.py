"""Episode-level metric aggregation."""
from __future__ import annotations
import numpy as np


def compute_episode_metrics(rewards, costs, h_values, violations, diag_summary):
    rewards = np.asarray(rewards, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    h_values = np.asarray(h_values, dtype=np.float64)
    viol = np.asarray(violations, dtype=np.float64)
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
