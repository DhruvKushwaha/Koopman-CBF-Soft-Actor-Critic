"""Velocity-norm barrier with quadratic feature augmentation."""
from __future__ import annotations
import numpy as np
from .barrier_base import SafetyConstraint


class VelocityNormBarrier(SafetyConstraint):
    name = "velocity_norm"

    def __init__(self, v_max: float, vel_indices):
        self.v_max = float(v_max)
        self.vel_indices = list(vel_indices)

    def value(self, raw_state, info):
        vs = np.asarray(raw_state)[self.vel_indices]
        return self.v_max ** 2 - float(np.sum(vs ** 2))

    def lifted_barrier_coeffs(self, z_dim: int, dim_y: int, n_rbf: int = 0, **kwargs):
        extra_start = dim_y + n_rbf
        needed = extra_start + len(self.vel_indices)
        if needed > z_dim:
            raise ValueError(
                f"VelocityNormBarrier: z_dim={z_dim} too small; need at least {needed} "
                f"(dim_y={dim_y} + n_rbf={n_rbf} + {len(self.vel_indices)} vel dims)."
            )
        c = np.zeros(z_dim, dtype=np.float64)
        for k in range(len(self.vel_indices)):
            c[extra_start + k] = -1.0
        d = self.v_max ** 2
        extra = {"quadratic_indices": list(self.vel_indices)}
        return c, float(d), extra

    def extra_features(self) -> dict:
        return {"quadratic_indices": list(self.vel_indices)}
