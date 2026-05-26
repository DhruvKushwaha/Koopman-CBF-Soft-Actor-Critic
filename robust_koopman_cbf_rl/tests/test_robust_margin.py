import numpy as np


def test_global_margin():
    from robust_koopman_cbf_rl.cbf.robust_margin import RobustMargin
    rm = RobustMargin(alpha=0.9, mode="global")
    rm.update(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
    rho = rm.get_margin()
    assert 0.85 <= rho <= 1.0


def test_cluster_margin_falls_back_to_global_with_few_points():
    from robust_koopman_cbf_rl.cbf.robust_margin import RobustMargin
    rm = RobustMargin(alpha=0.9, mode="cluster", n_clusters=4)
    rm.update(np.array([0.1, 0.5, 0.9]), zs=np.array([[0.0], [1.0], [2.0]]))
    rho = rm.get_margin(z=np.array([1.5]))
    assert rho >= 0.0
