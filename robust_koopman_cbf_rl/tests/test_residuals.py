import numpy as np
import pytest


def test_residuals_and_quantile_margin():
    from robust_koopman_cbf_rl.koopman.observables import RBFObservables
    from robust_koopman_cbf_rl.koopman.model import KoopmanModel
    from robust_koopman_cbf_rl.koopman.fit_edmd import fit_edmd
    from robust_koopman_cbf_rl.koopman.residuals import (
        compute_residuals, compute_robust_margin)

    rng = np.random.default_rng(0)
    Y = rng.normal(size=(300, 2))
    U = rng.normal(size=(300, 1))
    A_true = np.array([[0.9, 0.1], [0.0, 0.95]])
    B_true = np.array([[0.1], [0.2]])
    Yp = Y @ A_true.T + U @ B_true.T + 0.01 * rng.normal(size=(300, 2))

    obs = RBFObservables(dim_y=2, n_rbf=0, bandwidth=1.0, seed=0)
    obs.fit_centers(Y)
    A, B = fit_edmd(obs.lift_batch(Y), obs.lift_batch(Yp), U, reg=1e-6)
    model = KoopmanModel(obs, A, B)

    c = np.array([1.0, 0.0])
    deltas = compute_residuals(model, Y, U, Yp, c=c)
    assert deltas.shape == (300,)
    rho = compute_robust_margin(deltas, alpha=0.95)
    # Verify against numpy's own percentile
    expected = float(np.percentile(deltas, 95.0, method="linear"))
    assert np.isclose(rho, expected, rtol=1e-9)
