"""Stateful decomposition helpers."""

from __future__ import annotations

import numpy as np

from . import linalg, sparse


class SVD:
    """Reusable container for SVD factors."""

    def __init__(self, a, *, full_matrices: bool = False):
        self.U, self.s, self.Vt = linalg.svd(a, full_matrices=full_matrices)

    def solve(self, b, *, rcond: float = 1e-12):
        """Least-squares solve using pseudoinverse from SVD factors."""
        b_arr = np.asarray(b, dtype=np.float64)
        inv_s = np.where(np.abs(self.s) > rcond, 1.0 / self.s, 0.0)
        s_inv = np.diag(inv_s)
        return self.Vt.T @ s_inv @ self.U.T @ b_arr


class QR:
    """Reusable container for QR factors."""

    def __init__(self, a, *, mode: str = "reduced"):
        self.Q, self.R = linalg.qr(a, mode=mode)

    def solve(self, b):
        """Solve Ax=b using QR factors."""
        b_arr = np.asarray(b, dtype=np.float64)
        return np.linalg.solve(self.R, self.Q.T @ b_arr)


class SparseFactorized:
    """Wrapper around Eigen sparse factorization object."""

    def __init__(self, a, *, method: str = "auto"):
        self._impl = sparse.factorize(a, method=method)

    def solve(self, b):
        return self._impl.solve(np.asarray(b, dtype=np.float64))
