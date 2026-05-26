"""Wraps safety_gymnasium envs into a uniform interface with cost + velocity in info."""
from __future__ import annotations
import numpy as np
import gymnasium as gym


class SafetyGymnasiumWrapper(gym.Wrapper):
    """Exposes cost and an estimated forward velocity in the info dict."""

    def __init__(self, env, velocity_limit: float = 2.0):
        super().__init__(env)
        self.velocity_limit = float(velocity_limit)

    @staticmethod
    def _extract_velocity(env, info: dict) -> float:
        # safety_gymnasium MuJoCo locomotion envs expose qvel via the underlying task.
        for attr_chain in (
            ("task", "agent", "qvel"),
            ("unwrapped", "data", "qvel"),
            ("unwrapped", "task", "agent", "qvel"),
        ):
            obj = env
            ok = True
            for attr in attr_chain:
                if not hasattr(obj, attr):
                    ok = False
                    break
                obj = getattr(obj, attr)
            if ok and obj is not None:
                arr = np.asarray(obj)
                if arr.size > 0:
                    return float(arr[0])
        # Fall back to info dict
        return float(info.get("velocity", 0.0))

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        info = dict(info)
        info["velocity"] = self._extract_velocity(self.env, info)
        info["cost"] = float(info.get("cost", 0.0))
        return np.asarray(obs, dtype=np.float64), info

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            cost = float(info.get("cost", 0.0))
        elif len(out) == 6:
            obs, reward, cost, terminated, truncated, info = out
            cost = float(cost)
        else:
            obs, reward, done, info = out[:4]
            terminated, truncated, cost = bool(done), False, float(info.get("cost", 0.0))
        info = dict(info)
        info["cost"] = cost
        info["velocity"] = self._extract_velocity(self.env, info)
        return (
            np.asarray(obs, dtype=np.float64),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )


def make_safety_gym_env(env_id: str, velocity_limit: float = 2.0, seed: int = 0) -> SafetyGymnasiumWrapper:
    import safety_gymnasium
    env = safety_gymnasium.make(env_id)
    try:
        env.reset(seed=seed)
    except Exception:
        pass
    return SafetyGymnasiumWrapper(env, velocity_limit=velocity_limit)
