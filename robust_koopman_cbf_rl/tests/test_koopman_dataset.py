import numpy as np
from pathlib import Path


def test_dataset_collects_triples(tmp_path):
    from robust_koopman_cbf_rl.koopman.dataset import KoopmanDataset
    ds = KoopmanDataset(dim_y=4, dim_u=1)
    ds.add(np.zeros(4), np.array([0.5]), np.ones(4))
    ds.add(np.ones(4), np.array([-0.3]), 2 * np.ones(4))
    Y, U, Yp = ds.as_arrays()
    assert Y.shape == (2, 4) and U.shape == (2, 1) and Yp.shape == (2, 4)
    assert np.allclose(Y[1], 1.0) and np.allclose(Yp[1], 2.0)
    p = tmp_path / "ds.npz"
    ds.save(p)
    ds2 = KoopmanDataset.load(p)
    Y2, U2, Yp2 = ds2.as_arrays()
    np.testing.assert_allclose(Y, Y2)
    np.testing.assert_allclose(U, U2)
    np.testing.assert_allclose(Yp, Yp2)
