"""Pass-through filter and dummy model for unconstrained RL baselines."""
from __future__ import annotations
import numpy as np


class NullModel:
    """Dummy Koopman model — used when no safety filter is needed."""
    z_dim = 1

    class _Obs:
        dim_y = 1
    observables = _Obs()

    def lift(self, y: np.ndarray) -> np.ndarray:
        return np.zeros(1, dtype=np.float64)


class NullFilter:
    """Pass-through safety filter for unconstrained baselines.

    Implements the KCBFQPFilter interface so KCBFSACAgent/KCBFPPOAgent can be used
    without modification. No cbf_penalty_terms → CBF actor loss is automatically skipped.
    """

    def __init__(self, u_min: np.ndarray, u_max: np.ndarray):
        self.u_min = np.asarray(u_min, dtype=np.float64)
        self.u_max = np.asarray(u_max, dtype=np.float64)

    def project(self, z: np.ndarray, u_nom: np.ndarray):
        u_safe = np.clip(np.asarray(u_nom, dtype=np.float64), self.u_min, self.u_max)
        diag = {
            "h_value": float("nan"), "cbf_lhs": float("nan"), "cbf_rhs": float("nan"),
            "cbf_gap": float("nan"), "slack": 0.0, "correction_norm": 0.0,
            "intervention": False, "rho": 0.0,
        }
        return u_safe, diag
