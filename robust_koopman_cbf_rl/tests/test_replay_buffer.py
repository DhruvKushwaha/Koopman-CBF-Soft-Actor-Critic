import numpy as np

def test_kcbf_replay_buffer_stores_full_record():
    from robust_koopman_cbf_rl.agents.replay_buffer import KCBFReplayBuffer
    buf = KCBFReplayBuffer(capacity=10, dim_obs=4, dim_action=1, dim_z=8)
    buf.add(
        obs=np.zeros(4), z=np.zeros(8),
        action_nom=np.array([0.1]), action_safe=np.array([0.05]),
        reward=1.0, cost=0.0,
        next_obs=np.ones(4), next_z=np.ones(8),
        done=False,
        h_value=0.3, cbf_gap=0.05, slack=0.0, intervention=True,
    )
    batch = buf.sample(1, rng=np.random.default_rng(0))
    assert batch["action_safe"].shape == (1, 1)
    assert batch["action_nom"].shape == (1, 1)
    assert batch["intervention"].shape == (1,)
