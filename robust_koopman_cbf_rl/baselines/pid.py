"""Thin wrapper for safe_control_gym PID controller (when available)."""
from __future__ import annotations
import numpy as np


class PIDBaseline:
    def __init__(self, env, **pid_kwargs):
        from safe_control_gym.controllers.pid.pid import PID
        self.ctrl = PID(env=env, **pid_kwargs)

    def select_action(self, obs, info=None):
        a = self.ctrl.select_action(obs)
        return np.asarray(a, dtype=np.float64)
