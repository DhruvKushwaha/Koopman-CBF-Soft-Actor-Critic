import numpy as np

def test_rollout_buffer_stores_nom_safe_and_logprob():
    from robust_koopman_cbf_rl.agents.rollout_buffer import KCBFRolloutBuffer
    buf = KCBFRolloutBuffer(rollout_len=4, dim_obs=2, dim_action=1, dim_z=4)
    for t in range(4):
        buf.add(obs=np.zeros(2), z=np.zeros(4),
                action_nom=np.array([0.1]), action_safe=np.array([0.05]),
                logprob_nom=-0.3, value=0.5,
                reward=1.0, cost=0.0, done=False,
                h_value=0.2, cbf_gap=0.1, slack=0.0, intervention=False)
    buf.compute_advantages(last_value=0.0, gamma=0.99, lam=0.95)
    out = buf.batched()
    assert out["action_nom"].shape == (4, 1)
    assert out["action_safe"].shape == (4, 1)
    assert out["advantages"].shape == (4,)
    assert out["logprob_nom"].shape == (4,)
