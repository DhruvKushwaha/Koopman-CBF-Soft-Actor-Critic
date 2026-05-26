"""One-step and multi-step prediction error for Koopman models."""
from __future__ import annotations
import numpy as np


def compute_one_step_mse(model, Y, U, Yp) -> float:
    """MSE of lifted one-step predictions vs. lifted next states."""
    Z = model.lift_batch(np.asarray(Y, dtype=np.float64))
    U = np.asarray(U, dtype=np.float64)
    Zp_pred = Z @ model.A.T + U @ model.B.T
    Zp_true = model.lift_batch(np.asarray(Yp, dtype=np.float64))
    err = Zp_pred - Zp_true
    return float(np.mean(err ** 2))


def compute_multistep_mse(model, Y_traj, U_traj, horizons=(5, 10, 20, 50)) -> dict:
    """MSE over H-step rollouts starting from each timestep."""
    Y_traj = np.asarray(Y_traj, dtype=np.float64)
    U_traj = np.asarray(U_traj, dtype=np.float64)
    T = len(Y_traj)
    out: dict[int, float] = {}
    for H in horizons:
        if H >= T:
            continue
        errs = []
        for t in range(T - H):
            z = model.lift(Y_traj[t])
            for k in range(H):
                z = model.predict(z, U_traj[t + k])
            z_true = model.lift(Y_traj[t + H])
            errs.append(float(np.mean((z - z_true) ** 2)))
        out[H] = float(np.mean(errs)) if errs else float("nan")
    return out
