"""Off-policy replay buffer with both nominal and safe actions + filter diagnostics."""
from __future__ import annotations
import numpy as np


class KCBFReplayBuffer:
    def __init__(self, capacity: int, dim_obs: int, dim_action: int, dim_z: int):
        self.capacity = int(capacity)
        self.dim_obs = int(dim_obs)
        self.dim_action = int(dim_action)
        self.dim_z = int(dim_z)
        self._ptr = 0
        self._size = 0
        self.obs = np.zeros((capacity, dim_obs), dtype=np.float32)
        self.z = np.zeros((capacity, dim_z), dtype=np.float32)
        self.action_nom = np.zeros((capacity, dim_action), dtype=np.float32)
        self.action_safe = np.zeros((capacity, dim_action), dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.cost = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, dim_obs), dtype=np.float32)
        self.next_z = np.zeros((capacity, dim_z), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.h_value = np.zeros(capacity, dtype=np.float32)
        self.cbf_gap = np.zeros(capacity, dtype=np.float32)
        self.slack = np.zeros(capacity, dtype=np.float32)
        self.intervention = np.zeros(capacity, dtype=np.float32)

    def __len__(self):
        return self._size

    def add(self, obs, z, action_nom, action_safe, reward, cost,
            next_obs, next_z, done, h_value, cbf_gap, slack, intervention):
        i = self._ptr
        self.obs[i] = obs
        self.z[i] = z
        self.action_nom[i] = action_nom
        self.action_safe[i] = action_safe
        self.reward[i] = reward
        self.cost[i] = cost
        self.next_obs[i] = next_obs
        self.next_z[i] = next_z
        self.done[i] = float(done)
        self.h_value[i] = h_value
        self.cbf_gap[i] = cbf_gap
        self.slack[i] = slack
        self.intervention[i] = float(intervention)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict:
        idx = rng.integers(0, self._size, size=batch_size)
        return {
            "obs": self.obs[idx], "z": self.z[idx],
            "action_nom": self.action_nom[idx], "action_safe": self.action_safe[idx],
            "reward": self.reward[idx], "cost": self.cost[idx],
            "next_obs": self.next_obs[idx], "next_z": self.next_z[idx],
            "done": self.done[idx],
            "h_value": self.h_value[idx], "cbf_gap": self.cbf_gap[idx],
            "slack": self.slack[idx], "intervention": self.intervention[idx],
        }
