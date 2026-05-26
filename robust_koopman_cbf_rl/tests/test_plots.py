import numpy as np
import pandas as pd
import os


def _make_training_csv(tmp_path, name="log.csv"):
    path = tmp_path / name
    rows = [
        {"step": i * 100, "ep_return": float(i), "ep_cost": 0.1,
         "cbf_gap_mean": 0.5, "intervention_rate": 0.2}
        for i in range(20)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_plot_training_creates_file(tmp_path):
    from robust_koopman_cbf_rl.plots.plot_training import plot_training
    csv = _make_training_csv(tmp_path)
    out = str(tmp_path / "training.png")
    plot_training([csv], ["agent"], out)
    assert os.path.exists(out)


def test_plot_training_missing_columns_no_crash(tmp_path):
    from robust_koopman_cbf_rl.plots.plot_training import plot_training
    path = tmp_path / "sparse.csv"
    pd.DataFrame([{"step": 0, "ep_return": 1.0}]).to_csv(path, index=False)
    out = str(tmp_path / "sparse.png")
    plot_training([str(path)], ["sparse"], out)


def test_plot_trajectory_creates_file(tmp_path):
    from robust_koopman_cbf_rl.plots.plot_trajectory import plot_trajectory
    T, dim_y = 50, 4
    traj = {
        "states": np.random.randn(T, dim_y),
        "actions": np.random.randn(T, 1),
        "h_values": np.random.randn(T),
    }
    out = str(tmp_path / "traj.png")
    plot_trajectory(traj, state_names=["x", "x_dot", "theta", "theta_dot"], out_path=out)
    assert os.path.exists(out)


def test_compare_models_creates_file(tmp_path):
    from robust_koopman_cbf_rl.eval.compare_models import compare_models
    rows = [{"return": float(i), "total_cost": 0.1, "violation_rate": 0.0, "min_h_value": 0.5}
            for i in range(10)]
    p1 = tmp_path / "a.csv"; p2 = tmp_path / "b.csv"
    pd.DataFrame(rows).to_csv(p1, index=False)
    pd.DataFrame(rows).to_csv(p2, index=False)
    out = str(tmp_path / "compare.png")
    compare_models([str(p1), str(p2)], ["SAC", "PPO"], out)
    assert os.path.exists(out)
