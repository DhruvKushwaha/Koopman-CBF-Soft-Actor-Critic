"""CartPole linear barriers (split half-spaces for position and angle)."""
from __future__ import annotations
import numpy as np
from .barrier_base import SafetyConstraint


class CartPolePositionBarrier(SafetyConstraint):
    name = "cartpole_position"

    def __init__(self, x_max: float = 2.2, side: str = "right"):
        self.x_max = float(x_max)
        assert side in ("right", "left")
        self.side = side

    def value(self, raw_state, info):
        x = float(raw_state[0])
        return self.x_max - (x if self.side == "right" else -x)

    def lifted_barrier_coeffs(self, z_dim: int, dim_y: int = 4, **kwargs):
        c = np.zeros(z_dim, dtype=np.float64)
        c[0] = -1.0 if self.side == "right" else 1.0
        return c, float(self.x_max)


class CartPoleAngleBarrier(SafetyConstraint):
    name = "cartpole_angle"

    def __init__(self, theta_max: float = 0.14, side: str = "right"):
        self.theta_max = float(theta_max)
        assert side in ("right", "left")
        self.side = side

    def value(self, raw_state, info):
        th = float(raw_state[2])
        return self.theta_max - (th if self.side == "right" else -th)

    def lifted_barrier_coeffs(self, z_dim: int, dim_y: int = 4, **kwargs):
        c = np.zeros(z_dim, dtype=np.float64)
        c[2] = -1.0 if self.side == "right" else 1.0
        return c, float(self.theta_max)
