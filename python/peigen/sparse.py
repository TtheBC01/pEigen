"""Sparse matrix routines backed by Eigen."""

from __future__ import annotations

import numpy as np

from . import _core


def _require_scipy():
    try:
        import scipy.sparse as sp
    except ImportError as exc:  # pragma: no cover - exercised in tests
        raise ImportError(
            "Sparse routines require SciPy. Install with `pip install peigen[sparse]`."
        ) from exc
    return sp


def _as_2d_rhs(b):
    rhs = np.ascontiguousarray(np.asarray(b, dtype=np.float64))
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
    return _core.spmm(a, np.asarray(b, dtype=np.float64))


def spspmm(a, b):
    """Multiply sparse matrix `a` by sparse matrix `b`."""
    sp = _require_scipy()
    if not sp.issparse(a):
        a = sp.csc_matrix(a)
    if not sp.issparse(b):
        b = sp.csc_matrix(b)
    return _core.spspmm(a, b)


def solve(a, b, *, method: str = "auto", tol: float = 1e-8, maxiter: int | None = None):
    """Solve sparse linear system a x = b."""
    sp = _require_scipy()
    if not sp.issparse(a):
        a = sp.csc_matrix(a)
    rhs, squeezed = _as_2d_rhs(b)
    x = _core.sparse_solve(a, rhs, method, tol, 0 if maxiter is None else maxiter)
    return x[:, 0] if squeezed else x


def factorize(a, *, method: str = "auto"):
    """Factorize sparse matrix and return reusable solver object."""
    if method != "auto":
        raise ValueError("only method='auto' is currently supported")
    sp = _require_scipy()
    if not sp.issparse(a):
        a = sp.csc_matrix(a)
    return _core.sparse_factorize(a)


def to_dense(a):
    """Convert sparse matrix to dense ndarray."""
    return np.asarray(a.toarray(), dtype=np.float64)


def from_coo(data, row, col, shape):
    """Build sparse matrix from COO inputs."""
    sp = _require_scipy()
    return sp.coo_matrix((data, (row, col)), shape=shape).tocsc()
