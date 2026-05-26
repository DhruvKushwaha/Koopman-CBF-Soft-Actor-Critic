"""CBF-QP using known dynamics.

The caller supplies a callable `dynamics(x, u) -> x_next`. At each `project()` call this filter
finite-differences cᵀf(x, u) w.r.t. u at u=0 to extract the action-Jacobian `a = ∂(cᵀf)/∂u`, then
enforces the CBF condition cᵀ·x_next + d ≥ (1-η)(cᵀx + d) on the linearised next-step value.
No analytical (A, B) needed — works for any nonlinear f. State-side terms are evaluated exactly.

For safe-control-gym envs, wrap `env.symbolic.f_func` (CasADi) into a numpy `f(x, u)`. For
Safety-Gymnasium, supply a hand-written approximation since the underlying MuJoCo dynamics
are not exposed.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import csc_matrix
from qpsolvers import solve_qp


class PhysicalCBFFilter:
    def __init__(self, dynamics, dim_state, dim_action,
                 h_coeffs, eta, u_min, u_max, lam_slack: float = 1e6):
        self.f = dynamics
        self.dim_state = int(dim_state)
        self.dim_action = int(dim_action)
        self.c, self.d = h_coeffs[0].astype(np.float64), float(h_coeffs[1])
        self.eta = float(eta)
        self.u_min = np.asarray(u_min, dtype=np.float64)
        self.u_max = np.asarray(u_max, dtype=np.float64)
        self.lam_slack = float(lam_slack)

    def _b_from_action(self, x, u):
        x_next = self.f(x, u)
        return float(self.c @ x_next + self.d)

    def project(self, x, u_nom):
        x = np.asarray(x, dtype=np.float64)
        u_nom = np.asarray(u_nom, dtype=np.float64)
        # Finite difference to get a = ∂(c^T f)/∂u and offset b0.
        eps = 1e-4
        b0 = self._b_from_action(x, np.zeros(self.dim_action))
        a = np.zeros(self.dim_action)
        for i in range(self.dim_action):
            e = np.zeros(self.dim_action); e[i] = eps
            a[i] = (self._b_from_action(x, e) - b0) / eps
        h_x = float(self.c @ x + self.d)
        rhs = (1.0 - self.eta) * h_x - b0
        P = np.eye(self.dim_action + 1) * 2.0
        P[-1, -1] = 2.0 * self.lam_slack
        q = np.concatenate([-2.0 * u_nom, np.zeros(1)])
        G = np.concatenate([-a[None, :], -np.ones((1, 1))], axis=1)
        h_in = np.array([-rhs], dtype=np.float64)
        lb = np.concatenate([self.u_min, np.zeros(1)])
        ub = np.concatenate([self.u_max, np.array([np.inf])])
        sol = solve_qp(csc_matrix(P), q, G=csc_matrix(G), h=h_in, lb=lb, ub=ub, solver="osqp",
                       eps_abs=1e-2, eps_rel=1e-2)
        if sol is None:
            import warnings
            warnings.warn("PhysicalCBFFilter: QP infeasible, clamping to action bounds.")
            u_safe = np.clip(u_nom, self.u_min, self.u_max); xi = 0.0
        else:
            u_safe = sol[:self.dim_action]; xi = float(sol[-1])
        diag = {"cbf_gap": float(a @ u_safe + xi - rhs), "slack": xi}
        return u_safe.astype(np.float64), diag
