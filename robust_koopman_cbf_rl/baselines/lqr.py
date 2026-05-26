"""Thin wrapper for safe_control_gym LQR controller."""
from __future__ import annotations
import numpy as np


class LQRBaseline:
    def __init__(self, env, q_lqr=None, r_lqr=None, **lqr_kwargs):
        from safe_control_gym.controllers.lqr.lqr import LQR
        # LQR expects a factory callable, not an env instance; extract raw BenchmarkEnv
        raw_env = env._env if hasattr(env, '_env') else env
        # [1] broadcasts to full state/input diagonal via get_cost_weight_matrix
        self.ctrl = LQR(env_func=lambda: raw_env,
                        q_lqr=q_lqr or [1],
                        r_lqr=r_lqr or [0.1],
                        **lqr_kwargs)

    def select_action(self, obs, info=None):
        # strip goal-horizon appended dims (obs_goal_horizon > 0 gives nx+k obs)
        a = self.ctrl.select_action(obs[:self.ctrl.model.nx], info=info)
        return np.asarray(a, dtype=np.float64)
