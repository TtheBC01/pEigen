"""Dense linear algebra routines backed by Eigen."""

from __future__ import annotations

import numpy as np

from . import _core


def _as_2d_float64(a: np.ndarray | list[float] | list[list[float]]) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("input must be a 2D array")
    return arr


def _as_2d_for_factorization(a: np.ndarray | list[float] | list[list[float]]) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("input must be a 2D array")
    # Column-major layout is preferred for LAPACK factorization paths.
    return np.asfortranarray(arr)


def matmul(a, b):
    """Matrix multiplication for 2D dense arrays."""
    arr_a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    arr_b = np.ascontiguousarray(np.asarray(b, dtype=np.float64))
    if arr_a.ndim != 2 or arr_b.ndim != 2:
        raise ValueError("inputs must be 2D arrays")
    return _core.matmul(arr_a, arr_b)


def solve(a, b, *, assume_a: str = "gen", method: str = "auto"):
    """Solve a x = b for dense matrices."""
    if assume_a != "gen":
        raise ValueError("assume_a currently only supports 'gen'")
    lhs = _as_2d_float64(a)
    rhs = np.asarray(b, dtype=np.float64)
    if rhs.ndim == 1:
        rhs = rhs[:, None]
        squeezed = True
    elif rhs.ndim == 2:
        squeezed = False
    else:
        raise ValueError("b must be 1D or 2D")

    x = _core.solve(lhs, rhs, method)
    return x[:, 0] if squeezed else x


def qr(a, *, mode: str = "reduced"):
    """Compute QR decomposition of a dense matrix."""
    return _core.qr(_as_2d_for_factorization(a), mode)


def svd(a, *, full_matrices: bool = False, method: str = "auto"):
    """Compute singular value decomposition."""
    return _core.svd(_as_2d_for_factorization(a), full_matrices, method)


def svd_compute(a, *, full_matrices: bool = False, method: str = "auto") -> float:
    """Run SVD and return a scalar summary (for compute-only benchmarking)."""
    return _core.svd_compute(_as_2d_for_factorization(a), full_matrices, method)


def eigh(a, *, lower: bool = True, eigenvectors: bool = True, method: str = "auto"):
    """Compute eigenpairs of a symmetric/hermitian matrix."""
    return _core.eigh(_as_2d_for_factorization(a), lower, eigenvectors, method)


def eighvals(a, *, lower: bool = True, method: str = "auto"):
    """Compute eigenvalues of a symmetric/hermitian matrix."""
    # Staging to column-major is handled once in the extension.
    return _core.eigh(_as_2d_float64(a), lower, False, method)


def eigh_compute(a, *, lower: bool = True, method: str = "auto") -> float:
    """Run a symmetric eigenvalue solve and return a scalar summary (for benchmarking)."""
    return _core.eigh_compute(_as_2d_float64(a), lower, method)


def norm(a, ord=None, axis=None, keepdims: bool = False):
    """Norm wrapper with fast path for common Frobenius case."""
    arr = np.asarray(a, dtype=np.float64)
    if axis is None and ord is None and arr.ndim == 2:
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        out = _core.norm(arr)
        if keepdims:
            return np.asarray([[out]])
        return out
    return np.linalg.norm(arr, ord=ord, axis=axis, keepdims=keepdims)
