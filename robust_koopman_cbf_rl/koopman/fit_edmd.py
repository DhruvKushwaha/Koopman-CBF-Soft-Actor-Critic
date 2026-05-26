"""Ridge regression for EDMD: [A | B] via normal equations."""
from __future__ import annotations
import numpy as np
from scipy.linalg import solve


def fit_edmd(Z: np.ndarray, Zp: np.ndarray, U: np.ndarray, reg: float = 1e-6):
    """Solve min_{A,B} ||Zp - Z A^T - U B^T||_F^2 + reg ||[A|B]||^2.

    Returns A (z_dim, z_dim), B (z_dim, u_dim).
    """
    Z = np.asarray(Z, dtype=np.float64)
    Zp = np.asarray(Zp, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    n, z_dim = Z.shape
    u_dim = U.shape[1]
    G = np.concatenate([Z, U], axis=1)          # (n, z_dim+u_dim)
    gram = G.T @ G + reg * np.eye(z_dim + u_dim)  # (z_dim+u_dim, z_dim+u_dim)
    rhs = G.T @ Zp                               # (z_dim+u_dim, z_dim)
    coeff = solve(gram, rhs, assume_a="sym")     # (z_dim+u_dim, z_dim)
    A = coeff[:z_dim, :].T                       # (z_dim, z_dim)
    B = coeff[z_dim:, :].T                       # (z_dim, u_dim)
    return A.astype(np.float64), B.astype(np.float64)
