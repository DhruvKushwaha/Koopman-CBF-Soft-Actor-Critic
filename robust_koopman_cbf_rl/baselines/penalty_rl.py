"""Reward-penalty wrapper: r' = r - lam_c * cost."""
from __future__ import annotations


def penalize_reward(reward: float, cost: float, lam_c: float) -> float:
    return float(reward) - float(lam_c) * float(cost)


class RewardPenaltyMixin:
    """Mixin that wraps an env step to add cost penalty to the reward."""

    def __init__(self, *args, lam_c: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lam_c = float(lam_c)

    def shape_reward(self, reward, info):
        import warnings
        if "cost" not in info:
            warnings.warn("RewardPenaltyMixin: 'cost' key missing from info; treating as 0.")
        return penalize_reward(reward, float(info.get("cost", 0.0)), self.lam_c)
