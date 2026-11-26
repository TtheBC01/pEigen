"""Stateful decomposition helpers."""

from __future__ import annotations

import numpy as np

from . import linalg, sparse
from ._utils import as_real_float64


class SVD:
    """Reusable container for SVD factors."""

    def __init__(self, a, *, full_matrices: bool = False):
        self.U, self.s, self.Vt = linalg.svd(a, full_matrices=full_matrices)
        self.shape = (self.U.shape[0], self.Vt.shape[1])

    def solve(self, b, *, rcond: float | None = None):
        """Minimum-norm least-squares solve via the SVD pseudoinverse.

        Singular values ``<= rcond * max(s)`` are treated as zero. ``rcond`` is *relative* to the
        largest singular value, matching ``numpy.linalg.lstsq``; the default ``None`` uses
        ``eps * max(M, N)``.
        """
        b_arr = as_real_float64(b, "b")
        m, n = self.shape
        if b_arr.ndim not in (1, 2) or b_arr.shape[0] != m:
            raise ValueError(f"b must have shape ({m},) or ({m}, k)")
        if rcond is None:
            rcond = np.finfo(np.float64).eps * max(m, n)
        k = self.s.shape[0]
        # Only the leading k singular vectors matter; this also handles full_matrices=True.
        u, vt = self.U[:, :k], self.Vt[:k, :]
        smax = self.s.max() if k else 0.0
        inv_s = np.zeros_like(self.s)
        keep = self.s > rcond * smax
        inv_s[keep] = 1.0 / self.s[keep]
        y = u.T @ b_arr
        y = inv_s[:, None] * y if y.ndim == 2 else inv_s * y
        return vt.T @ y


class QR:
    """Reusable container for QR factors."""

    def __init__(self, a, *, mode: str = "reduced"):
        self.Q, self.R = linalg.qr(a, mode=mode)
        self.shape = (self.Q.shape[0], self.R.shape[1])

    def solve(self, b):
        """Solve ``A x = b`` for square ``A``, or the least-squares problem ``min ||A x - b||`` for
        tall ``A`` (``m > n``), from the QR factors of either ``mode``.

        ``A`` must have full column rank. Wide ``A`` (``m < n``) raises ``ValueError``: QR without
        column pivoting does not give the minimum-norm solution, so use ``SVD.solve`` instead.
        """
        b_arr = as_real_float64(b, "b")
        m, n = self.shape
        if b_arr.ndim not in (1, 2) or b_arr.shape[0] != m:
            raise ValueError(f"b must have shape ({m},) or ({m}, k)")
        if m < n:
            raise ValueError(
                f"QR.solve needs a square or tall matrix, got shape ({m}, {n}); "
                "use SVD.solve for the minimum-norm solution of an underdetermined system"
            )
        # mode="complete" returns Q (m, m) and R (m, n); only the leading n columns of Q and rows
        # of R take part (the rest of R is zero).
        return np.linalg.solve(self.R[:n, :], self.Q[:, :n].T @ b_arr)


# Same class as ``peigen.sparse.factorize`` returns; re-exported here for discoverability.
SparseFactorized = sparse.SparseFactorized
SparseCholesky = sparse.SparseCholesky
