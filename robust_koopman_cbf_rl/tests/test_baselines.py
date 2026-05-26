import numpy as np

def test_reward_penalty_applies_cost():
    from robust_koopman_cbf_rl.baselines.penalty_rl import penalize_reward
    r = penalize_reward(reward=1.0, cost=0.5, lam_c=2.0)
    assert np.isclose(r, 1.0 - 2.0 * 0.5)

def test_lagrangian_dual_increases_on_violation():
    from robust_koopman_cbf_rl.baselines.lagrangian_rl import LagrangianDual
    d = LagrangianDual(init_value=0.0, lr=0.1, budget=0.0)
    d.update(mean_cost=0.5)
    assert d.value > 0.0
    v1 = d.value
    d.update(mean_cost=0.5)
    assert d.value > v1

def test_lagrangian_dual_decreases_on_slack():
    from robust_koopman_cbf_rl.baselines.lagrangian_rl import LagrangianDual
    d = LagrangianDual(init_value=1.0, lr=0.1, budget=0.5)
    d.update(mean_cost=0.0)
    assert d.value < 1.0

def test_physical_cbf_filter_returns_safe_action():
    from robust_koopman_cbf_rl.baselines.physical_cbf_qp import PhysicalCBFFilter
    # System: x_{t+1} = x + 0.05 * [u, 0]; constraint: -x[0] + 2.4 >= 0 (i.e. x[0] <= 2.4)
    # h_coeffs = (c, d): h(x) = c @ x + d = -x[0] + 2.4
    def f(x, u): return x + 0.05 * np.array([u[0], 0.0])
    flt = PhysicalCBFFilter(dynamics=f, dim_state=2, dim_action=1,
                            h_coeffs=(np.array([-1.0, 0.0]), 2.4),
                            eta=0.5, u_min=np.array([-1.0]), u_max=np.array([1.0]))
    # State very close to boundary, nominal pushes further out — filter must intervene.
    x = np.array([2.38, 0.0])
    u_nom = np.array([1.0])  # would push x[0] to 2.43, violating h(x_next) >= (1-eta)*h(x)
    u_safe, diag = flt.project(x, u_nom)
    assert u_safe.shape == (1,)
    assert "cbf_gap" in diag
    assert diag["cbf_gap"] >= -1e-4, f"CBF constraint violated: gap={diag['cbf_gap']}"
    # The safe action must differ from nominal (filter must have intervened)
    assert u_safe[0] < u_nom[0] - 1e-3, f"Expected filter to reduce u; got u_safe={u_safe[0]:.4f}"
