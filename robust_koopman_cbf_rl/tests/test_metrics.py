import numpy as np

def test_episode_metrics_from_synthetic_data():
    from robust_koopman_cbf_rl.utils.metrics import compute_episode_metrics
    rewards = [1.0, 1.0, 0.5]
    costs = [0.0, 1.0, 0.0]
    h_vals = [0.5, -0.1, 0.3]   # one violation
    viol = [0.0, 1.0, 0.0]
    diag = {"intervention_rate": 0.33, "slack_rate": 0.0,
            "cbf_gap_min": 0.0, "correction_norm_mean": 0.0}
    m = compute_episode_metrics(rewards, costs, h_vals, viol, diag)
    assert m["return"] == 2.5
    assert m["total_cost"] == 1.0
    assert m["violation_rate"] == 1 / 3
    assert m["episode_violation"] == 1.0
    assert m["intervention_rate"] == 0.33
    assert m["min_h_value"] == -0.1
