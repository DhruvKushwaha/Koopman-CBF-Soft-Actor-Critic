"""RBF observable dictionary: z = [y, rbf_i(y), optional quadratic features]."""
from __future__ import annotations
import numpy as np


class RBFObservables:
    def __init__(self, dim_y: int, n_rbf: int, bandwidth: float = 1.0,
                 extra_quadratic_indices: list[int] | None = None, seed: int = 0):
        self.dim_y = int(dim_y)
        self.n_rbf = int(n_rbf)
        self.bandwidth = float(bandwidth)
        self.extra_quadratic_indices = list(extra_quadratic_indices or [])
        self.seed = int(seed)
        self.centers: np.ndarray | None = None

    @property
    def z_dim(self) -> int:
        return self.dim_y + self.n_rbf + len(self.extra_quadratic_indices)

    def fit_centers(self, Y: np.ndarray) -> None:
        Y = np.asarray(Y, dtype=np.float64)
        assert Y.shape[1] == self.dim_y
        if self.n_rbf == 0:
            self.centers = np.zeros((0, self.dim_y), dtype=np.float64)
            return
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(Y.shape[0], size=self.n_rbf, replace=(Y.shape[0] < self.n_rbf))
        self.centers = Y[idx].copy()

    def _rbf(self, Y: np.ndarray) -> np.ndarray:
        if self.n_rbf == 0:
            return np.zeros((Y.shape[0], 0), dtype=np.float64)
        # Y: (N, dim_y); centers: (n_rbf, dim_y). Output: (N, n_rbf)
        diff = Y[:, None, :] - self.centers[None, :, :]
        sq = np.sum(diff * diff, axis=2)
        return np.exp(-sq / (2.0 * self.bandwidth ** 2))

    def _quadratic(self, Y: np.ndarray) -> np.ndarray:
        if not self.extra_quadratic_indices:
            return np.zeros((Y.shape[0], 0), dtype=np.float64)
        return Y[:, self.extra_quadratic_indices] ** 2

    def lift_batch(self, Y: np.ndarray) -> np.ndarray:
        assert self.centers is not None, "Call fit_centers(Y) first."
        Y = np.asarray(Y, dtype=np.float64).reshape(-1, self.dim_y)
        parts = [Y, self._rbf(Y)]
        q = self._quadratic(Y)
        if q.shape[1] > 0:
            parts.append(q)
        return np.concatenate(parts, axis=1)

    def lift(self, y: np.ndarray) -> np.ndarray:
        return self.lift_batch(np.asarray(y).reshape(1, -1))[0]
