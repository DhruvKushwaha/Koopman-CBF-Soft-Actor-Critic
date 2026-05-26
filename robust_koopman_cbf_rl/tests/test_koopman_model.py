import numpy as np


def test_koopman_model_predict_and_save_load(tmp_path):
    from robust_koopman_cbf_rl.koopman.observables import RBFObservables
    from robust_koopman_cbf_rl.koopman.model import KoopmanModel

    rng = np.random.default_rng(0)
    Y = rng.normal(size=(200, 3))
    obs = RBFObservables(dim_y=3, n_rbf=4, bandwidth=1.0, seed=0)
    obs.fit_centers(Y)
    A = np.eye(obs.z_dim)
    B = np.ones((obs.z_dim, 2))
    m = KoopmanModel(observables=obs, A=A, B=B)
    y = np.array([0.1, -0.2, 0.3])
    z = m.lift(y)
    assert z.shape == (obs.z_dim,)
    u = np.array([0.5, -0.5])
    zp = m.predict(z, u)
    np.testing.assert_allclose(zp, A @ z + B @ u)
    path = tmp_path / "m.npz"
    m.save(path)
    m2 = KoopmanModel.load(path)
    np.testing.assert_allclose(m2.A, A)
    np.testing.assert_allclose(m2.B, B)
    np.testing.assert_allclose(m2.lift(y), z)
