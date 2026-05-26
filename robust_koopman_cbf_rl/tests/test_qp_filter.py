import numpy as np
import pytest


def _make_filter(eta=0.5, lam_slack=1e3):
    pytest.importorskip("qpsolvers")
    from robust_koopman_cbf_rl.koopman.observables import RBFObservables
    from robust_koopman_cbf_rl.koopman.model import KoopmanModel
    from robust_koopman_cbf_rl.cbf.cartpole_barriers import CartPolePositionBarrier
    from robust_koopman_cbf_rl.cbf.robust_margin import RobustMargin
    from robust_koopman_cbf_rl.cbf.qp_filter import KCBFQPFilter

    obs = RBFObservables(dim_y=4, n_rbf=0, bandwidth=1.0, seed=0)
    obs.fit_centers(np.zeros((1, 4)))
    A = np.eye(4)
    B = np.zeros((4, 1))
    m = KoopmanModel(obs, A, B)
    bar = CartPolePositionBarrier(x_max=2.2, side="right")
    rm = RobustMargin(alpha=0.95, mode="global")
    rm.update(np.zeros(10))
    flt = KCBFQPFilter(m, bar, rm, eta=eta, lam_slack=lam_slack,
                       u_min=np.array([-10.0]), u_max=np.array([10.0]))
    return m, flt


def test_qp_passes_through_safe_nominal():
    m, flt = _make_filter()
    y = np.array([0.0, 0.0, 0.0, 0.0])
    z = m.lift(y)
    u_nom = np.array([0.5])
    u_safe, diag = flt.project(z, u_nom)
    np.testing.assert_allclose(u_safe, u_nom, atol=1e-5)
    assert diag["slack"] < 1e-5


def test_qp_modifies_unsafe_nominal():
    pytest.importorskip("qpsolvers")
    from robust_koopman_cbf_rl.koopman.observables import RBFObservables
    from robust_koopman_cbf_rl.koopman.model import KoopmanModel
    from robust_koopman_cbf_rl.cbf.cartpole_barriers import CartPolePositionBarrier
    from robust_koopman_cbf_rl.cbf.robust_margin import RobustMargin
    from robust_koopman_cbf_rl.cbf.qp_filter import KCBFQPFilter

    obs = RBFObservables(dim_y=4, n_rbf=0, bandwidth=1.0, seed=0)
    obs.fit_centers(np.zeros((1, 4)))
    # x_{t+1} = x_t + x_dot_t;  x_dot_{t+1} = x_dot_t + u
    A = np.eye(4)
    A[0, 1] = 1.0
    B = np.zeros((4, 1))
    B[1, 0] = 1.0
    m = KoopmanModel(obs, A, B)
    bar = CartPolePositionBarrier(x_max=2.2, side="right")
    rm = RobustMargin(alpha=0.95, mode="global")
    rm.update(np.zeros(10))
    flt = KCBFQPFilter(m, bar, rm, eta=0.5, lam_slack=1e6,
                       u_min=np.array([-10.0]), u_max=np.array([10.0]))
    y = np.array([2.1, 0.5, 0.0, 0.0])  # near barrier, moving right
    z = m.lift(y)
    u_nom = np.array([5.0])              # would push further right
    u_safe, diag = flt.project(z, u_nom)
    # CBF constraint must hold: lhs + slack >= rhs (equivalently cbf_gap >= 0)
    assert diag["cbf_gap"] + 1e-6 >= 0
    assert u_safe[0] <= u_nom[0]         # action reduced


def test_diagnostics_buffer_records_steps():
    from robust_koopman_cbf_rl.cbf.diagnostics import DiagnosticsBuffer
    buf = DiagnosticsBuffer()
    buf.log(h_value=0.5, cbf_gap=0.1, slack=0.0, correction_norm=0.0, intervention=False)
    buf.log(h_value=-0.1, cbf_gap=0.0, slack=0.05, correction_norm=0.3, intervention=True)
    stats = buf.summary()
    assert stats["intervention_rate"] == 0.5
    assert stats["slack_rate"] == 0.5
