"""Wraps safe_control_gym BenchmarkEnv into gymnasium API with augmented info dict."""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SafeControlGymWrapper(gym.Env):
    """Adapts BenchmarkEnv to gymnasium step/reset API and augments info dict.

    safe_control_gym uses the OLD gym API:
      - reset() returns (obs, info)        [gymnasium-compatible already]
      - step()  returns (obs, rew, done, info)  [4-tuple, not 5-tuple]

    This wrapper:
    1. Converts step() to the 5-tuple gymnasium API (obs, rew, terminated, truncated, info).
    2. Augments info to always contain:
       - raw_state         (np.ndarray): full physical state from env.state
       - reference_state   (np.ndarray): goal/reference state (env.X_GOAL)
       - tracking_error    (np.ndarray): raw_state - reference_state
       - constraint_values (np.ndarray): constraint function values (zeros if unconstrained)
       - cost              (float):      1.0 if any constraint violated, else 0.0
    3. Exposes gymnasium Box observation_space and action_space.
    """

    metadata = {"render_modes": []}

    def __init__(self, benchmark_env):
        self._env = benchmark_env
        self.observation_space = spaces.Box(
            low=np.asarray(self._env.observation_space.low, dtype=np.float64),
            high=np.asarray(self._env.observation_space.high, dtype=np.float64),
            dtype=np.float64,
        )
        self.action_space = spaces.Box(
            low=np.asarray(self._env.action_space.low, dtype=np.float64),
            high=np.asarray(self._env.action_space.high, dtype=np.float64),
            dtype=np.float64,
        )

    def _get_reference_state(self) -> np.ndarray:
        """Extracts the reference/goal state from the underlying env."""
        ref = getattr(self._env, "X_GOAL", None)
        if ref is None:
            ref = getattr(self._env, "goal", None)
        if ref is None:
            return np.zeros(self._env.state_dim, dtype=np.float64)
        ref = np.asarray(ref, dtype=np.float64)
        # For trajectory tracking, X_GOAL is shape (N, state_dim); use current step index.
        if ref.ndim == 2:
            step = getattr(self._env, "ctrl_step_counter", 0)
            idx = min(step, ref.shape[0] - 1)
            ref = ref[idx]
        return ref.ravel()

    def _build_info(self, base_info: dict) -> dict:
        """Augments the base info dict with required keys."""
        info = dict(base_info) if base_info else {}

        # raw_state: the full physical state set on env.state after reset/step
        raw = getattr(self._env, "state", None)
        if raw is not None:
            info["raw_state"] = np.asarray(raw, dtype=np.float64).ravel()
        else:
            info["raw_state"] = np.zeros(self._env.state_dim, dtype=np.float64)

        # reference_state: goal state from X_GOAL (or zeros if unavailable)
        info["reference_state"] = self._get_reference_state()

        # tracking_error: element-wise difference, trimmed to the shorter dimension
        raw_state = info["raw_state"]
        ref_state = info["reference_state"]
        min_dim = min(raw_state.shape[0], ref_state.shape[0])
        info["tracking_error"] = raw_state[:min_dim] - ref_state[:min_dim]

        # constraint_values: from info if constraints were configured, else zeros
        cv = base_info.get("constraint_values") if base_info else None
        if cv is not None:
            info["constraint_values"] = np.asarray(cv, dtype=np.float64)
        else:
            info["constraint_values"] = np.zeros(0, dtype=np.float64)

        # cost: 1.0 if any constraint is violated, otherwise 0.0
        violation = base_info.get("constraint_violation", 0) if base_info else 0
        info["cost"] = float(bool(violation))

        return info

    def reset(self, *, seed=None, options=None):
        """Reset the environment.

        Returns:
            obs (np.ndarray): Initial observation.
            info (dict): Augmented info dict containing required keys.
        """
        # safe_control_gym reset() accepts seed kwarg directly
        if seed is not None:
            try:
                self._env.seed(seed)
            except Exception:
                pass
        out = self._env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, base_info = out
        else:
            obs, base_info = out, {}
        if not isinstance(base_info, dict):
            base_info = {}
        return np.asarray(obs, dtype=np.float64), self._build_info(base_info)

    def step(self, action):
        """Step the environment.

        Returns:
            obs (np.ndarray):    Next observation.
            reward (float):      Scalar reward.
            terminated (bool):   True if episode ended due to task completion/failure.
            truncated (bool):    True if episode ended due to time limit.
            info (dict):         Augmented info dict containing required keys.
        """
        action = np.asarray(action, dtype=np.float64)
        out = self._env.step(action)

        if len(out) == 5:
            # Gymnasium 5-tuple (unlikely for safe_control_gym but handle gracefully)
            obs, reward, terminated, truncated, base_info = out
        elif len(out) == 4:
            # Old gym 4-tuple: (obs, rew, done, info)
            obs, reward, done, base_info = out
            # Distinguish true termination from time-limit truncation
            time_limit_truncated = base_info.get("TimeLimit.truncated", False) if isinstance(base_info, dict) else False
            if time_limit_truncated:
                terminated = False
                truncated = True
            else:
                terminated = bool(done)
                truncated = False
        else:
            obs, reward, done = out[:3]
            base_info = {}
            terminated, truncated = bool(done), False

        if not isinstance(base_info, dict):
            base_info = {}

        return (
            np.asarray(obs, dtype=np.float64),
            float(reward),
            bool(terminated),
            bool(truncated),
            self._build_info(base_info),
        )

    def close(self):
        """Close the underlying environment."""
        try:
            self._env.close()
        except Exception:
            pass

    def render(self):
        """Delegate rendering to the underlying environment."""
        return self._env.render()


def make_safe_control_gym_env(
    task: str, task_config: dict, seed: int = 0
) -> SafeControlGymWrapper:
    """Create a SafeControlGymWrapper around a safe_control_gym environment.

    Args:
        task (str): The registered environment ID (e.g. 'cartpole', 'quadrotor').
        task_config (dict): Keyword arguments forwarded to the environment constructor
            (e.g. task='stabilization', ctrl_freq=50, episode_len_sec=5).
        seed (int): Random seed used to initialize the environment.

    Returns:
        SafeControlGymWrapper: A gymnasium-compatible wrapper with augmented info.
    """
    from safe_control_gym.utils.registration import make as scg_make

    cfg = dict(task_config)
    env = scg_make(task, **cfg)
    try:
        env.seed(seed)
    except Exception:
        pass
    return SafeControlGymWrapper(env)
