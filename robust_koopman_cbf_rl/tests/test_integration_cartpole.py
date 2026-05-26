import numpy as np
import pytest
from pathlib import Path

pytest.importorskip("safe_control_gym")

_CONFIGS = Path(__file__).parent.parent / "configs"


@pytest.mark.slow
def test_kcbf_sac_cartpole_5000_steps_satisfies_theorem(tmp_path):
    """Train SAC + KCBF for 5000 steps on CartPole stab; check theorem and violations."""
    from robust_koopman_cbf_rl.train.train_koopman import main as train_kp
    from robust_koopman_cbf_rl.train.train_sac_kcbf import main as train_sac
    env_cfg = str(_CONFIGS / "env_cartpole_stab.yaml")
    k_cfg = str(_CONFIGS / "koopman.yaml")
    model_path = tmp_path / "k.npz"
    train_kp(env_cfg, k_cfg, str(model_path))
    log_dir = tmp_path / "logs"
    train_sac(env_cfg, str(_CONFIGS / "sac_kcbf.yaml"),
              str(model_path), str(log_dir), total_steps=5000)
    import pandas as pd
    df = pd.read_csv(log_dir / "sac_kcbf.csv")
    # >= 95% of logged episodes must have non-negative cbf_gap_min
    ok = (df["cbf_gap_min"] >= -1e-4).mean()
    assert ok >= 0.95, f"CBF theorem satisfied in {ok:.2%} of logged episodes"


def test_theorem_check_inline():
    """Direct symbolic check: after filter, c^T (A z + B u_safe) + d − (1−η)(c^T z + d) − ρ + ξ ≥ 0."""
    from robust_koopman_cbf_rl.koopman.observables import RBFObservables
    from robust_koopman_cbf_rl.koopman.model import KoopmanModel
    from robust_koopman_cbf_rl.cbf.cartpole_barriers import CartPolePositionBarrier
    from robust_koopman_cbf_rl.cbf.robust_margin import RobustMargin
    from robust_koopman_cbf_rl.cbf.qp_filter import KCBFQPFilter

    obs = RBFObservables(dim_y=4, n_rbf=0, bandwidth=1.0, seed=0)
    obs.fit_centers(np.zeros((1, 4)))
    A = np.eye(4); A[0, 1] = 1.0
    B = np.zeros((4, 1)); B[1, 0] = 1.0
    m = KoopmanModel(obs, A, B)
    bar = CartPolePositionBarrier(x_max=2.2, side="right")
    rm = RobustMargin(alpha=0.95); rm.update(np.array([0.01, 0.02, 0.03]))
    flt = KCBFQPFilter(m, bar, rm, eta=0.5, lam_slack=1e6,
                      u_min=np.array([-10.0]), u_max=np.array([10.0]))

    rng = np.random.default_rng(0)
    for _ in range(50):
        y = rng.normal(scale=0.5, size=4)
        y[0] = np.clip(y[0], -2.0, 2.19)
        z = m.lift(y)
        u_nom = rng.normal(scale=2.0, size=1)
        u_safe, diag = flt.project(z, u_nom)
        result = bar.lifted_barrier_coeffs(z_dim=z.shape[0])
        c, d = result[0], result[1]
        lhs = float(c @ (A @ z + B @ u_safe) + d)
        rhs = (1.0 - flt.eta) * float(c @ z + d) + diag["rho"]
        assert lhs + diag["slack"] + 1e-6 >= rhs, \
            f"theorem violated: lhs={lhs}, rhs={rhs}, slack={diag['slack']}"
