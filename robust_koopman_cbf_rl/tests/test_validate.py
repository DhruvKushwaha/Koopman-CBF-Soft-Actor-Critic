import numpy as np


def test_one_step_and_multistep_mse_on_perfect_linear():
    from robust_koopman_cbf_rl.koopman.observables import RBFObservables
    from robust_koopman_cbf_rl.koopman.model import KoopmanModel
    from robust_koopman_cbf_rl.koopman.validate import (
        compute_one_step_mse, compute_multistep_mse)

    rng = np.random.default_rng(0)
    Y = rng.normal(size=(500, 2)).astype(np.float64)
    obs = RBFObservables(dim_y=2, n_rbf=0, bandwidth=1.0, seed=0)
    obs.fit_centers(Y)
    A = np.array([[0.9, 0.1], [0.0, 0.95]])
    B = np.zeros((2, 1))
    m = KoopmanModel(obs, A, B)
    # Build a self-consistent trajectory under the model.
    U = np.zeros((100, 1))
    traj = [Y[0]]
    z = m.lift(traj[-1])
    for t in range(99):
        z = m.predict(z, U[t])
        traj.append(z[:2])
    Y_traj = np.stack(traj)
    assert compute_one_step_mse(m, Y_traj[:-1], U[:-1], Y_traj[1:]) < 1e-12
    ms = compute_multistep_mse(m, Y_traj, U, horizons=[5, 10, 20])
    for h in (5, 10, 20):
        assert ms[h] < 1e-10
