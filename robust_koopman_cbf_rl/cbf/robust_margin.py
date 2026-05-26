"""Empirical residual quantile margin (global and cluster-based)."""
from __future__ import annotations
import numpy as np


class RobustMargin:
    def __init__(self, alpha: float = 0.95, mode: str = "global",
                 n_clusters: int = 8):
        assert mode in ("global", "cluster")
        self.alpha = float(alpha)
        self.mode = mode
        self.n_clusters = int(n_clusters)
        self._deltas: list[np.ndarray] = []
        self._zs: list[np.ndarray] = []
        self._centroids: np.ndarray | None = None
        self._cluster_rho: np.ndarray | None = None
        self._global_rho: float = 0.0

    def update(self, deltas: np.ndarray, zs: np.ndarray | None = None) -> None:
        deltas = np.asarray(deltas, dtype=np.float64).ravel()
        self._deltas.append(deltas)
        if zs is not None:
            self._zs.append(np.asarray(zs, dtype=np.float64))
        all_d = np.concatenate(self._deltas)
        self._global_rho = float(np.percentile(all_d, 100.0 * self.alpha, method="linear"))
        if self.mode == "cluster" and self._zs:
            self._fit_clusters()

    def _fit_clusters(self) -> None:
        Z = np.concatenate(self._zs, axis=0)
        D = np.concatenate(self._deltas, axis=0)
        rng = np.random.default_rng(0)
        k = min(self.n_clusters, Z.shape[0])
        idx = rng.choice(Z.shape[0], size=k, replace=False)
        centroids = Z[idx].copy()
        for _ in range(20):
            dists = np.sum((Z[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            assign = np.argmin(dists, axis=1)
            new_cent = centroids.copy()
            for j in range(k):
                if np.any(assign == j):
                    new_cent[j] = Z[assign == j].mean(axis=0)
            if np.allclose(new_cent, centroids):
                break
            centroids = new_cent
        self._centroids = centroids
        rhos = np.empty(k, dtype=np.float64)
        for j in range(k):
            mask = assign == j
            if np.any(mask):
                rhos[j] = float(np.percentile(D[mask], 100.0 * self.alpha, method="linear"))
            else:
                rhos[j] = self._global_rho
        self._cluster_rho = rhos

    def get_margin(self, z: np.ndarray | None = None) -> float:
        if self.mode == "global" or self._centroids is None or z is None:
            return self._global_rho
        dists = np.sum((self._centroids - np.asarray(z, dtype=np.float64)) ** 2, axis=1)
        return float(self._cluster_rho[int(np.argmin(dists))])
