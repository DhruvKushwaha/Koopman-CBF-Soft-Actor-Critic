import numpy as np
import pytest


def test_cartpole_wrapper_info_keys():
    from robust_koopman_cbf_rl.envs.safe_control_gym_wrapper import make_safe_control_gym_env
    env = make_safe_control_gym_env(task="cartpole", task_config={
        "task": "stabilization",
        "ctrl_freq": 50,
        "episode_len_sec": 5,
    }, seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    required = {"raw_state", "reference_state", "tracking_error",
                "constraint_values", "cost"}
    assert required.issubset(info.keys())
    a = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(a)
    assert "raw_state" in info
    assert isinstance(info["cost"], float)
    env.close()
