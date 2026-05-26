"""Lagrangian dual variable update: λ <- max(0, λ + lr * (mean_cost - budget))."""
from __future__ import annotations


class LagrangianDual:
    def __init__(self, init_value: float = 0.0, lr: float = 0.01, budget: float = 0.0):
        self.value = float(init_value)
        self.lr = float(lr)
        self.budget = float(budget)

    def update(self, mean_cost: float) -> float:
        self.value = max(0.0, self.value + self.lr * (float(mean_cost) - self.budget))
        return self.value
