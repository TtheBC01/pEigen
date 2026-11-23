"""Sparse matrix routines backed by Eigen."""

from __future__ import annotations

import numpy as np

from . import _core
from ._utils import as_real_float64


def _require_scipy():
    try:
        import scipy.sparse as sp
    except ImportError as exc:  # pragma: no cover - exercised in tests
        raise ImportError(
            "Sparse routines require SciPy. Install with `pip install peigen[sparse]`."
        ) from exc
    return sp


def _as_2d_rhs(b):
    rhs = np.ascontiguousarray(as_real_float64(b, "b"))
    if rhs.ndim == 1:
        return rhs[:, None], True
    if rhs.ndim == 2:
        return rhs, False
    raise ValueError("rhs must be a 1D or 2D array")


def spmm(a, b):
    """Multiply sparse matrix `a` by dense matrix `b`."""
    sp = _require_scipy()
    if not sp.issparse(a):
        a = sp.csc_matrix(a)
    return _core.spmm(a, as_real_float64(b, "b"))


def spspmm(a, b):
    """Multiply sparse matrix `a` by sparse matrix `b`."""
    sp = _require_scipy()
    if not sp.issparse(a):
        a = sp.csc_matrix(a)
    if not sp.issparse(b):
        b = sp.csc_matrix(b)
    return _core.spspmm(a, b)


def solve(
    a,
    b,
    *,
    method: str = "auto",
    tol: float = 1e-8,
    maxiter: int | None = None,
    preconditioner: str = "none",
    ilu_fill_factor: int = 10,
    ilu_drop_tol: float = 1e-4,
):
    """Solve sparse linear system a x = b."""
    sp = _require_scipy()
    if not sp.issparse(a):
        a = sp.csc_matrix(a)
    rhs, squeezed = _as_2d_rhs(b)
    x = _core.sparse_solve(
        a,
        rhs,
        method,
        tol,
        0 if maxiter is None else maxiter,
        preconditioner,
        ilu_fill_factor,
        ilu_drop_tol,
    )
    return x[:, 0] if squeezed else x


def solve_stats(
    a,
    b,
    *,
    method: str = "cg",
    tol: float = 1e-8,
    maxiter: int | None = None,
    preconditioner: str = "none",
    ilu_fill_factor: int = 10,
    ilu_drop_tol: float = 1e-4,
):
    """Return iteration count and estimated error for an iterative CG solve."""
    sp = _require_scipy()
    if not sp.issparse(a):
        a = sp.csc_matrix(a)
    rhs, squeezed = _as_2d_rhs(b)
    if not squeezed and rhs.shape[1] != 1:
        raise ValueError("solve_stats requires a single RHS column")
    return _core.sparse_solve_stats(
        a,
        rhs,
        method,
        tol,
        0 if maxiter is None else maxiter,
        preconditioner,
        ilu_fill_factor,
        ilu_drop_tol,
    )


class SparseFactorized:
    """Reusable sparse LU factorization (Eigen ``SparseLU``).

    Factorizes ``a`` once; :meth:`solve` can then be called many times with different right-hand
    sides. ``b`` may be 1D (returns 1D) or 2D with one system per column (returns 2D).
    """

    def __init__(self, a, *, method: str = "auto"):
        if method != "auto":
            raise ValueError("only method='auto' is currently supported")
        sp = _require_scipy()
        if not sp.issparse(a):
            a = sp.csc_matrix(a)
        self._impl = _core.sparse_factorize(a)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self._impl.shape)

    def solve(self, b):
        rhs, squeezed = _as_2d_rhs(b)
        x = self._impl.solve(rhs)
        return x[:, 0] if squeezed else x


def factorize(a, *, method: str = "auto") -> SparseFactorized:
    """Factorize sparse matrix and return reusable solver object."""
    return SparseFactorized(a, method=method)


def to_dense(a):
    """Convert sparse matrix to dense ndarray."""
    return np.asarray(a.toarray(), dtype=np.float64)


def from_coo(data, row, col, shape):
    """Build sparse matrix from COO inputs."""
    sp = _require_scipy()
    return sp.coo_matrix((data, (row, col)), shape=shape).tocsc()
