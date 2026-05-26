"""On-policy rollout buffer with nominal+safe actions and GAE."""
from __future__ import annotations
import numpy as np


class KCBFRolloutBuffer:
    def __init__(self, rollout_len: int, dim_obs: int, dim_action: int, dim_z: int):
        self.T = int(rollout_len)
        self.obs = np.zeros((self.T, dim_obs), dtype=np.float32)
        self.z = np.zeros((self.T, dim_z), dtype=np.float32)
        self.action_nom = np.zeros((self.T, dim_action), dtype=np.float32)
        self.action_safe = np.zeros((self.T, dim_action), dtype=np.float32)
        self.logprob_nom = np.zeros(self.T, dtype=np.float32)
        self.values = np.zeros(self.T, dtype=np.float32)
        self.rewards = np.zeros(self.T, dtype=np.float32)
        self.costs = np.zeros(self.T, dtype=np.float32)
        self.dones = np.zeros(self.T, dtype=np.float32)
        self.h_value = np.zeros(self.T, dtype=np.float32)
        self.cbf_gap = np.zeros(self.T, dtype=np.float32)
        self.slack = np.zeros(self.T, dtype=np.float32)
        self.intervention = np.zeros(self.T, dtype=np.float32)
        self.advantages = np.zeros(self.T, dtype=np.float32)
        self.returns = np.zeros(self.T, dtype=np.float32)
        self._t = 0

    def add(self, obs, z, action_nom, action_safe, logprob_nom, value,
            reward, cost, done, h_value, cbf_gap, slack, intervention):
        t = self._t
        self.obs[t] = obs; self.z[t] = z
        self.action_nom[t] = action_nom
        self.action_safe[t] = action_safe
        self.logprob_nom[t] = logprob_nom
        self.values[t] = value
        self.rewards[t] = reward
        self.costs[t] = cost
        self.dones[t] = float(done)
        self.h_value[t] = h_value
        self.cbf_gap[t] = cbf_gap
        self.slack[t] = slack
        self.intervention[t] = float(intervention)
        self._t += 1

    def compute_advantages(self, last_value: float, gamma: float, lam: float):
        adv = 0.0
        for t in reversed(range(self.T)):
            not_done = 1.0 - self.dones[t]
            next_v = last_value if t == self.T - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * not_done * next_v - self.values[t]
            adv = delta + gamma * lam * not_done * adv
            self.advantages[t] = adv
        self.returns = self.advantages + self.values

    def batched(self) -> dict:
        adv = self.advantages
        adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
        return {
            "obs": self.obs, "z": self.z,
            "action_nom": self.action_nom, "action_safe": self.action_safe,
            "logprob_nom": self.logprob_nom,
            "advantages": adv_norm, "returns": self.returns,
            "values": self.values,
        }

    def reset(self):
        self._t = 0
