import numpy as np
import pytest

safety_gymnasium = pytest.importorskip("safety_gymnasium")


def test_halfcheetah_velocity_cost():
    from robust_koopman_cbf_rl.envs.safety_gymnasium_wrapper import make_safety_gym_env
    env = make_safety_gym_env("SafetyHalfCheetahVelocity-v1", velocity_limit=2.0, seed=0)
    obs, info = env.reset(seed=0)
    assert obs.ndim == 1
    a = env.action_space.sample()
    obs, r, term, trunc, info = env.step(a)
    assert "cost" in info
    assert "velocity" in info
    assert isinstance(info["cost"], float)
    assert info["cost"] >= 0.0
    env.close()
