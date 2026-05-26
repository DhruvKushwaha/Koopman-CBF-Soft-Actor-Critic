"""Per-step residual r_t = z_{t+1} - (A z_t + B u_t); projected: delta_t = |c^T r_t|."""
from __future__ import annotations
import numpy as np


def compute_residuals(model, Y, U, Yp, c: np.ndarray) -> np.ndarray:
    """Returns delta_t = |c^T r_t| for each timestep."""
    Z = model.lift_batch(np.asarray(Y, dtype=np.float64))
    Zp = model.lift_batch(np.asarray(Yp, dtype=np.float64))
    U = np.asarray(U, dtype=np.float64)
    Zp_pred = Z @ model.A.T + U @ model.B.T
    R = Zp - Zp_pred                            # (N, z_dim)
    c = np.asarray(c, dtype=np.float64)
    proj = R @ c                                # (N,)
    return np.abs(proj).astype(np.float64)


def compute_robust_margin(deltas: np.ndarray, alpha: float = 0.95) -> float:
    """Empirical quantile of projected residuals."""
    deltas = np.asarray(deltas, dtype=np.float64)
    return float(np.percentile(deltas, 100.0 * alpha, method="linear"))
