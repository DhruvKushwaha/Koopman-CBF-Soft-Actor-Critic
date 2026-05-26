"""Collects (y_t, u_t, y_{t+1}) transitions for EDMD fitting."""
from __future__ import annotations
from pathlib import Path
import numpy as np


class KoopmanDataset:
    def __init__(self, dim_y: int, dim_u: int):
        self.dim_y = int(dim_y)
        self.dim_u = int(dim_u)
        self._y: list[np.ndarray] = []
        self._u: list[np.ndarray] = []
        self._yp: list[np.ndarray] = []

    def add(self, y, u, y_next):
        y = np.asarray(y, dtype=np.float64).reshape(self.dim_y)
        u = np.asarray(u, dtype=np.float64).reshape(self.dim_u)
        yp = np.asarray(y_next, dtype=np.float64).reshape(self.dim_y)
        self._y.append(y)
        self._u.append(u)
        self._yp.append(yp)

    def __len__(self):
        return len(self._y)

    def as_arrays(self):
        Y = np.stack(self._y) if self._y else np.zeros((0, self.dim_y))
        U = np.stack(self._u) if self._u else np.zeros((0, self.dim_u))
        Yp = np.stack(self._yp) if self._yp else np.zeros((0, self.dim_y))
        return Y, U, Yp

    def save(self, path):
        Y, U, Yp = self.as_arrays()
        np.savez(Path(path), Y=Y, U=U, Yp=Yp, dim_y=self.dim_y, dim_u=self.dim_u)

    @classmethod
    def load(cls, path):
        npz = np.load(Path(path))
        ds = cls(int(npz["dim_y"]), int(npz["dim_u"]))
        Y, U, Yp = npz["Y"], npz["U"], npz["Yp"]
        for i in range(len(Y)):
            ds.add(Y[i], U[i], Yp[i])
        return ds
