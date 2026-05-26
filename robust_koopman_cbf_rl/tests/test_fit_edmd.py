import numpy as np


def test_edmd_recovers_linear_system():
    from robust_koopman_cbf_rl.koopman.fit_edmd import fit_edmd
    rng = np.random.default_rng(0)
    A_true = np.array([[0.9, 0.1], [-0.05, 0.95]])
    B_true = np.array([[0.1], [0.2]])
    n = 5000
    Z = rng.normal(size=(n, 2))
    U = rng.normal(size=(n, 1))
    Zp = Z @ A_true.T + U @ B_true.T
    A, B = fit_edmd(Z, Zp, U, reg=1e-8)
    np.testing.assert_allclose(A, A_true, atol=1e-3)
    np.testing.assert_allclose(B, B_true, atol=1e-3)
