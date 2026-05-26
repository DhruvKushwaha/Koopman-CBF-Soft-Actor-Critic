"""KoopmanModel: combines RBFObservables with linear (A, B) dynamics."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .observables import RBFObservables


class KoopmanModel:
    def __init__(self, observables: RBFObservables, A: np.ndarray, B: np.ndarray):
        self.observables = observables
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        assert self.A.shape[0] == self.A.shape[1] == observables.z_dim
        assert self.B.shape[0] == observables.z_dim

    @property
    def z_dim(self) -> int:
        return self.observables.z_dim

    def lift(self, y: np.ndarray) -> np.ndarray:
        return self.observables.lift(y)

    def lift_batch(self, Y: np.ndarray) -> np.ndarray:
        return self.observables.lift_batch(Y)

    def predict(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.A @ np.asarray(z, dtype=np.float64) + self.B @ np.asarray(u, dtype=np.float64)

    def save(self, path) -> None:
        obs = self.observables
        centers = obs.centers if obs.centers is not None else np.zeros((0, obs.dim_y))
        np.savez(
            Path(path),
            A=self.A, B=self.B,
            centers=centers,
            dim_y=obs.dim_y, n_rbf=obs.n_rbf,
            bandwidth=obs.bandwidth,
            extra_quad=np.asarray(obs.extra_quadratic_indices, dtype=np.int64),
            seed=obs.seed,
        )

    @classmethod
    def load(cls, path) -> "KoopmanModel":
        npz = np.load(Path(path))
        obs = RBFObservables(
            dim_y=int(npz["dim_y"]),
            n_rbf=int(npz["n_rbf"]),
            bandwidth=float(npz["bandwidth"]),
            extra_quadratic_indices=list(npz["extra_quad"].astype(int)),
            seed=int(npz["seed"]),
        )
        obs.centers = npz["centers"]
        return cls(observables=obs, A=npz["A"], B=npz["B"])
