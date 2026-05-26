"""Quadrotor 2D / 3D linear barriers."""
from __future__ import annotations
import numpy as np
from .barrier_base import SafetyConstraint

# State layout for Quadrotor 2D: [x, x_dot, z, z_dot, theta, theta_dot]
_Z_IDX    = 2
_ZDOT_IDX = 3


class Quadrotor2DAltitudeBarrier(SafetyConstraint):
    name = "quad2d_altitude"

    def __init__(self, z_min: float, z_max: float, side: str = "upper"):
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        assert side in ("upper", "lower")
        self.side = side

    def value(self, raw_state, info):
        z = float(raw_state[2])
        return (self.z_max - z) if self.side == "upper" else (z - self.z_min)

    def lifted_barrier_coeffs(self, z_dim: int, dim_y: int = 6, state_index: int = 2, **kwargs):
        c = np.zeros(z_dim, dtype=np.float64)
        if self.side == "upper":
            c[state_index] = -1.0
            d = self.z_max
        else:
            c[state_index] = 1.0
            d = -self.z_min
        return c, float(d)


class Quadrotor2DPitchBarrier(SafetyConstraint):
    name = "quad2d_pitch"

    def __init__(self, theta_max: float, side: str = "right"):
        self.theta_max = float(theta_max)
        assert side in ("right", "left")
        self.side = side

    def value(self, raw_state, info):
        th = float(raw_state[4])
        return self.theta_max - (th if self.side == "right" else -th)

    def lifted_barrier_coeffs(self, z_dim: int, dim_y: int = 6, **kwargs):
        c = np.zeros(z_dim, dtype=np.float64)
        c[4] = -1.0 if self.side == "right" else 1.0
        return c, float(self.theta_max)


class Quadrotor2DCompositeAltitudeBarrier(SafetyConstraint):
    """Relative-degree-1 altitude barrier for Quadrotor 2D.

    h(z, z_dot) = alpha*(z - z_min) + beta*z_dot

    The velocity term beta*z_dot makes the barrier relative-degree-1: thrust
    directly controls z_dot (B[3,:] ≈ 79x larger than B[2,:]), so the QP
    filter has real control authority instead of near-zero a_cbf.

    Choosing alpha=1, beta=0.5 means the boundary h=0 corresponds to
    a maximum safe descent rate of 2*(z - z_min) m/s — slower near the floor,
    faster when high.  With init_z ∈ [0.3, 2.0] and init_z_dot ∈ [-1, 1],
    the initial state may be unsafe (h < 0) when z is low and z_dot is
    negative, but the QP can recover in a few steps via thrust.
    """

    name = "quad2d_composite_altitude"

    def __init__(self, z_min: float, z_max: float = 2.0,
                 alpha: float = 1.0, beta: float = 0.5):
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.alpha = float(alpha)
        self.beta  = float(beta)

    def value(self, raw_state, info):
        z     = float(raw_state[_Z_IDX])
        z_dot = float(raw_state[_ZDOT_IDX])
        return self.alpha * (z - self.z_min) + self.beta * z_dot

    def lifted_barrier_coeffs(self, z_dim: int, dim_y: int = 6, **kwargs):
        c = np.zeros(z_dim, dtype=np.float64)
        c[_Z_IDX]    = self.alpha
        c[_ZDOT_IDX] = self.beta
        d = -self.alpha * self.z_min   # h = c^T z + d = alpha*(z - z_min) + beta*z_dot
        return c, float(d)


class Quadrotor3DPositionBarrier(SafetyConstraint):
    name = "quad3d_position"

    def __init__(self, axis_index: int, lo: float, hi: float, side: str = "upper"):
        self.axis_index = int(axis_index)
        self.lo = float(lo)
        self.hi = float(hi)
        assert side in ("upper", "lower")
        self.side = side

    def value(self, raw_state, info):
        x = float(raw_state[self.axis_index])
        return (self.hi - x) if self.side == "upper" else (x - self.lo)

    def lifted_barrier_coeffs(self, z_dim: int, dim_y: int = 12, **kwargs):
        c = np.zeros(z_dim, dtype=np.float64)
        if self.side == "upper":
            c[self.axis_index] = -1.0
            d = self.hi
        else:
            c[self.axis_index] = 1.0
            d = -self.lo
        return c, float(d)
