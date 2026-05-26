"""Abstract barrier API."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class SafetyConstraint(ABC):
    """h(y) >= 0 in raw state space; lifted to c^T z + d in Koopman space."""

    name: str = "barrier"

    @abstractmethod
    def value(self, raw_state: np.ndarray, info: dict) -> float:
        """Return h(y); positive means safe."""

    def label(self, raw_state: np.ndarray, info: dict) -> str:
        return "safe" if self.value(raw_state, info) >= 0 else "unsafe"

    @abstractmethod
    def lifted_barrier_coeffs(self, z_dim: int, dim_y: int, **kwargs):
        """Return (c, d) or (c, d, extra) defining h_K(z) = c^T z + d."""

    def extra_features(self) -> dict:
        """Optional features the barrier needs in the lifted state, e.g.
        {'quadratic_indices': [...]}. Default: none."""
        return {}
