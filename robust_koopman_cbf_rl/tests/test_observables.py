import numpy as np


def test_rbf_observables_shape_and_identity_prefix():
    from robust_koopman_cbf_rl.koopman.observables import RBFObservables
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(500, 4))
    obs = RBFObservables(dim_y=4, n_rbf=20, bandwidth=1.0,
                        extra_quadratic_indices=None, seed=0)
    obs.fit_centers(Y)
    z = obs.lift(Y[0])
    assert z.shape == (4 + 20,)
    np.testing.assert_allclose(z[:4], Y[0])


def test_rbf_observables_with_quadratic_features():
    from robust_koopman_cbf_rl.koopman.observables import RBFObservables
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(100, 3))
    obs = RBFObservables(dim_y=3, n_rbf=5, bandwidth=1.0,
                        extra_quadratic_indices=[0, 1], seed=0)
    obs.fit_centers(Y)
    z = obs.lift(np.array([2.0, -3.0, 0.5]))
    assert z.shape == (3 + 5 + 2,)
    assert np.isclose(z[-2], 4.0)   # 2.0^2
    assert np.isclose(z[-1], 9.0)   # (-3.0)^2


def test_batch_lift():
    from robust_koopman_cbf_rl.koopman.observables import RBFObservables
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(50, 2))
    obs = RBFObservables(dim_y=2, n_rbf=4, bandwidth=0.8, seed=0)
    obs.fit_centers(Y)
    Z = obs.lift_batch(Y)
    assert Z.shape == (50, 2 + 4)
    np.testing.assert_allclose(Z[:, :2], Y)


def test_zero_rbf_case():
    from robust_koopman_cbf_rl.koopman.observables import RBFObservables
    Y = np.eye(4)
    obs = RBFObservables(dim_y=4, n_rbf=0, bandwidth=1.0, seed=0)
    obs.fit_centers(Y)
    z = obs.lift(Y[0])
    assert z.shape == (4,)
    np.testing.assert_allclose(z, Y[0])
