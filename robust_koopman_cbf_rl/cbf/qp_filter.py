"""KCBF QP safety filter: project nominal action to safe action via OSQP."""
from __future__ import annotations
import numpy as np
from scipy.sparse import csc_matrix
from qpsolvers import solve_qp


class KCBFQPFilter:
    """Solves: min ||u - u_nom||^2 + lam_slack * xi^2
                s.t. a^T u + xi >= b,
                     u_min <= u <= u_max,
                     xi >= 0
       where a = B^T c,  b = (1-eta)(c^T z + d) + rho - c^T A z - d.
    """

    def __init__(self, koopman_model, barrier, robust_margin,
                 eta: float, lam_slack: float,
                 u_min: np.ndarray, u_max: np.ndarray,
                 intervention_eps: float = 1e-5):
        self.model = koopman_model
        self.barrier = barrier
        self.robust_margin = robust_margin
        self.eta = float(eta)
        self.lam_slack = float(lam_slack)
        self.u_min = np.asarray(u_min, dtype=np.float64)
        self.u_max = np.asarray(u_max, dtype=np.float64)
        self.intervention_eps = float(intervention_eps)
        # Validate that barriers requiring extra quadratic features (e.g. VelocityNormBarrier)
        # match the extra_quadratic_indices baked into the Koopman observables at training time.
        if hasattr(barrier, "extra_features"):
            expected = list(barrier.extra_features().get("quadratic_indices", []))
            actual = list(getattr(koopman_model.observables, "extra_quadratic_indices", []))
            if expected != actual:
                raise ValueError(
                    f"Barrier quadratic_indices {expected} do not match "
                    f"model observables.extra_quadratic_indices {actual}. "
                    "Retrain the Koopman model with the same barrier's extra_features."
                )

    def _barrier_coeffs(self, z_dim: int):
        # Pass dim_y and n_rbf so barriers that use quadratic augmentation
        # (e.g. VelocityNormBarrier) can locate their features correctly.
        out = self.barrier.lifted_barrier_coeffs(
            z_dim=z_dim,
            dim_y=self.model.observables.dim_y,
            n_rbf=self.model.observables.n_rbf,
        )
        if isinstance(out, tuple) and len(out) == 3:
            c, d, _extra = out
        else:
            c, d = out
        return np.asarray(c, dtype=np.float64), float(d)

    def project(self, z: np.ndarray, u_nom: np.ndarray):
        z = np.asarray(z, dtype=np.float64)
        u_nom = np.asarray(u_nom, dtype=np.float64).ravel()
        A, B = self.model.A, self.model.B
        c, d = self._barrier_coeffs(z.shape[0])
        h_val = float(c @ z + d)
        rho = self.robust_margin.get_margin(z=z)

        # a_cbf = B^T c  (u_dim,)
        a = (B.T @ c).astype(np.float64).ravel()
        # b_cbf = (1-eta) h_val + rho - c^T (A z) - d
        b = (1.0 - self.eta) * h_val + rho - float(c @ (A @ z)) - d

        u_dim = u_nom.size
        # QP variables: x = [u (u_dim), xi (1)]
        P = np.zeros((u_dim + 1, u_dim + 1), dtype=np.float64)
        P[:u_dim, :u_dim] = 2.0 * np.eye(u_dim)
        P[u_dim, u_dim] = 2.0 * self.lam_slack
        q = np.concatenate([-2.0 * u_nom, np.zeros(1)])

        # Constraint: a^T u + xi >= b  ->  -a u - xi <= -b
        G_mat = np.zeros((1, u_dim + 1), dtype=np.float64)
        G_mat[0, :u_dim] = -a
        G_mat[0, u_dim] = -1.0
        h_ineq = np.array([-b], dtype=np.float64)

        lb = np.concatenate([self.u_min, np.zeros(1)])
        ub = np.concatenate([self.u_max, np.array([np.inf])])

        x = solve_qp(csc_matrix(P), q, G=csc_matrix(G_mat), h=h_ineq, lb=lb, ub=ub, solver="osqp",
                     eps_abs=1e-2, eps_rel=1e-2)

        if x is None:
            u_safe = np.clip(u_nom, self.u_min, self.u_max)
            xi = max(0.0, b - float(a @ u_safe))
        else:
            u_safe = x[:u_dim]
            xi = float(x[u_dim])

        cbf_gap = float(a @ u_safe + xi - b)
        correction_norm = float(np.linalg.norm(u_safe - u_nom))
        diag = {
            "h_value": h_val,
            "cbf_lhs": float(c @ (A @ z + B @ u_safe) + d),
            "cbf_rhs": float((1.0 - self.eta) * h_val + rho),
            "cbf_gap": cbf_gap,
            "slack": xi,
            "correction_norm": correction_norm,
            "intervention": correction_norm > self.intervention_eps,
            "rho": rho,
        }
        return u_safe.astype(np.float64), diag

    def project_batch(self, z_batch: np.ndarray, u_nom_batch: np.ndarray) -> np.ndarray:
        """Vectorized analytic projection of a batch of actions onto the CBF halfspace.

        For a single affine CBF constraint a^T u >= b (+ box bounds), the exact solution
        when u_min <= u_nom <= u_max is:
          - if a^T u_nom >= b: u_safe = u_nom  (already safe)
          - otherwise:         u_safe = clip(u_nom + λ*a, u_min, u_max)
                               where λ = (b - a^T u_nom) / (||a||^2)

        The clipping is exact for 1-D actions; for higher dimensions it is an approximation
        that may leave the box constraint active on some dimensions — sufficient for the SAC
        TD-target evaluation (no gradient needed) but not as tight as a full OSQP solve.

        All our current barriers (CartPolePositionBarrier, VelocityNormBarrier) produce
        state-independent (c, d), so the constraint coefficients are computed once for the
        batch using z_batch[0] as a representative sample for z_dim.
        """
        z_batch = np.asarray(z_batch, dtype=np.float64)
        u_nom_batch = np.asarray(u_nom_batch, dtype=np.float64)
        N = z_batch.shape[0]
        A, B = self.model.A, self.model.B
        c, d = self._barrier_coeffs(z_batch.shape[1])          # state-independent c, d
        a_cbf = (B.T @ c).ravel()                               # (u_dim,)
        a_norm_sq = float(a_cbf @ a_cbf) + 1e-12

        h_vals = z_batch @ c + d                                # (N,)
        AZ_c = (z_batch @ A.T) @ c                              # (N,)
        rho = float(self.robust_margin.get_margin(z=None))      # scalar (global mode)
        b_vals = (1.0 - self.eta) * h_vals + rho - AZ_c - d    # (N,)

        a_dot_u = u_nom_batch @ a_cbf                           # (N,)
        deficit = b_vals - a_dot_u                              # (N,)
        lam = np.maximum(0.0, deficit) / a_norm_sq             # (N,)
        u_proj = u_nom_batch + lam[:, None] * a_cbf[None, :]   # (N, u_dim)
        return np.clip(u_proj, self.u_min, self.u_max).astype(np.float32)

    def cbf_penalty_terms(self, z_batch, u_batch):
        """Return torch tensors (a_cbf [N, u_dim], b_cbf [N]) for actor CBF penalty."""
        import torch
        import numpy as np
        A_t = torch.as_tensor(self.model.A, dtype=z_batch.dtype, device=z_batch.device)
        B_t = torch.as_tensor(self.model.B, dtype=z_batch.dtype, device=z_batch.device)
        z_dim = z_batch.shape[1]
        c_np, d = self._barrier_coeffs(z_dim)
        c_t = torch.as_tensor(c_np, dtype=z_batch.dtype, device=z_batch.device)
        # Global margin used for batch penalty; per-sample cluster margin not supported here.
        rho = float(self.robust_margin.get_margin(z=None))
        h_z = z_batch @ c_t + d                           # (N,)
        Az = z_batch @ A_t.T                              # (N, z_dim)
        a_cbf = (B_t.T @ c_t).unsqueeze(0).expand(z_batch.shape[0], -1)  # (N, u_dim)
        b_cbf = (1.0 - self.eta) * h_z + rho - (Az @ c_t) - d            # (N,)
        return a_cbf, b_cbf
