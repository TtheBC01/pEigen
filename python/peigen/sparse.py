"""Sparse matrix routines backed by Eigen."""

from __future__ import annotations

import warnings

import numpy as np

from . import _core
from ._utils import as_real_float64


def _require_scipy():
    try:
        import scipy.sparse as sp
    except ImportError as exc:  # pragma: no cover - exercised in tests
        raise ImportError(
            "Sparse routines require SciPy. Install with `pip install peigentux[sparse]`."
        ) from exc
    return sp


def _as_2d_rhs(b):
    rhs = np.ascontiguousarray(as_real_float64(b, "b"))
    if rhs.ndim == 1:
        return rhs[:, None], True
    if rhs.ndim == 2:
        return rhs, False
    raise ValueError("rhs must be a 1D or 2D array")


def _warn_cg_ilu(method: str, preconditioner: str) -> None:
    if method == "cg" and preconditioner == "ilu":
        warnings.warn(
            "preconditioner='ilu' with method='cg' is deprecated and will be removed in pEigen 3.0: "
            "IncompleteLUT is not a symmetric preconditioner, so CG can fail to converge with it "
            "(even on a 2D Laplacian at the default ilu_fill_factor). "
            "Use preconditioner='ichol' (incomplete Cholesky) instead.",
            DeprecationWarning,
            stacklevel=3,
        )


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
    """Solve sparse linear system a x = b.

    ``method`` is one of ``"auto"``/``"lu"`` (SparseLU), ``"cholesky"`` (SimplicialLDLT; ``a`` must
    be symmetric, only its lower triangle is read), ``"cg"`` or ``"bicgstab"``.

    For the iterative methods, ``tol`` is the target relative residual ``||b - a x|| / ||b||``.
    After the solver stops, the true residual is recomputed; if it misses ``tol`` (by more than a
    factor of 10) the solver restarts from its current iterate, and ``RuntimeError`` is raised if
    that doesn't recover. ``RuntimeError`` is also raised when ``maxiter`` is reached.

    ``preconditioner`` (``method="cg"`` only) is ``"none"``, ``"jacobi"`` (diagonal) or
    ``"ichol"`` (incomplete Cholesky). ``"ilu"`` (IncompleteLUT, tuned by ``ilu_fill_factor`` and
    ``ilu_drop_tol``) is deprecated for CG because it is not symmetric.
    """
    sp = _require_scipy()
    if not sp.issparse(a):
        a = sp.csc_matrix(a)
    if method == "cholesky":
        if preconditioner != "none":
            raise ValueError("preconditioner is only supported for method='cg'")
        return SparseCholesky(a).solve(b)
    _warn_cg_ilu(method, preconditioner)
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
    """Run a CG solve and return ``{"iterations", "error", "residual"}``.

    ``iterations`` counts all CG iterations (including any restart), ``error`` is CG's own residual
    estimate and ``residual`` is the true relative residual ``||b - a x|| / ||b||``. Takes the same
    arguments as :func:`solve` and raises ``RuntimeError`` in the same cases.
    """
    sp = _require_scipy()
    if not sp.issparse(a):
        a = sp.csc_matrix(a)
    _warn_cg_ilu(method, preconditioner)
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


class SparseCholesky:
    """Reusable sparse factorization of a symmetric matrix (Eigen ``SimplicialLDLT/LLT`` or
    ``MultifrontalLDLT``).

    Computes ``P (A + shift*I) P^T = L D L^T`` (``mode="ldlt"``, default) or ``= L L^T``
    (``mode="llt"``) with a fill-reducing permutation ``P`` (``ordering="amd"``, default, or
    ``"natural"``). Only one triangle of ``a`` is read (``lower=True`` reads the lower triangle), so
    passing just that triangle is fine.

    ``mode="multifrontal"`` (experimental) factors the matrix as a tree of dense frontal matrices
    with level-3 updates, which is much faster on high-fill problems such as 3D meshes. ``D`` has
    1x1 and 2x2 blocks chosen by threshold pivoting, so it is also stable on symmetric *indefinite*
    matrices (e.g. KKT / saddle-point systems). Its ``P`` also depends on the values, and
    ``factor_nnz`` includes explicit zeros of merged fronts. The first solve with several
    right-hand sides after each factorization builds a supernodal copy of ``L`` for a blocked
    kernel (a one-time cost of roughly a third to two thirds of the factorization, plus the memory
    of a second ``L``); later many-RHS solves reuse it. Single-RHS solves never build it.

    ``mode="llt"`` requires ``A + shift*I`` to be positive definite. ``mode="ldlt"`` also factorizes
    some indefinite matrices, but it does not pivot for stability, so it is only reliable for
    symmetric positive (or negative) definite matrices; use ``mode="multifrontal"`` for indefinite
    input.

    Raises ``numpy.linalg.LinAlgError`` if the factorization breaks down (LLT on a matrix that is
    not positive definite, or a zero or non-finite pivot).
    """

    def __init__(self, a, *, mode: str = "ldlt", ordering: str = "amd", lower: bool = True, shift: float = 0.0):
        self._lower = bool(lower)
        self._impl = _core.SparseCholesky(self._prepare(a), mode, ordering, float(shift))

    def _prepare(self, a):
        sp = _require_scipy()
        if not sp.issparse(a):
            a = sp.csc_matrix(a)
        # Eigen reads the lower triangle; the upper triangle of A is the lower triangle of A.T.
        return a if self._lower else a.T

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self._impl.shape)

    @property
    def mode(self) -> str:
        return self._impl.mode

    @property
    def ordering(self) -> str:
        return self._impl.ordering

    @property
    def shift(self) -> float:
        return self._impl.shift

    @property
    def factor_nnz(self) -> int:
        """Number of stored nonzeros in the factor ``L`` (a measure of fill-in)."""
        return int(self._impl.factor_nnz)

    @property
    def inertia(self) -> tuple[int, int, int]:
        """``(positive, negative, zero)`` eigenvalue counts of ``A + shift*I``, read off the pivots of
        ``D`` (Sylvester's law of inertia)."""
        return tuple(int(v) for v in self._impl.inertia)

    def solve(self, b):
        """Solve ``(A + shift*I) x = b``; ``b`` may be 1D or 2D (one system per column)."""
        rhs = as_real_float64(b, "b")
        if rhs.ndim == 1:
            return self._impl.solve(rhs[:, None])[:, 0]
        if rhs.ndim == 2:
            return self._impl.solve(rhs)
        raise ValueError("rhs must be a 1D or 2D array")

    def refactorize(self, a, *, shift: float | None = None) -> "SparseCholesky":
        """Recompute the numeric factorization for new values with the *same* sparsity pattern.

        Reuses the symbolic analysis (ordering and elimination tree), which is typically much
        cheaper than a fresh :func:`cholesky`. ``shift=None`` keeps the current shift. Raises
        ``ValueError`` if the sparsity pattern differs from the original matrix.
        """
        self._impl.refactorize(self._prepare(a), None if shift is None else float(shift))
        return self

    def slogdet(self) -> tuple[float, float]:
        """Return ``(sign, logabsdet)`` of ``A + shift*I``, like :func:`numpy.linalg.slogdet`."""
        return tuple(self._impl.slogdet())

    def logdet(self) -> float:
        """Return ``log(det(A + shift*I))``; raises ``ValueError`` if the determinant is not positive."""
        sign, logabsdet = self.slogdet()
        if sign <= 0:
            raise ValueError("determinant is not positive; use slogdet()")
        return logabsdet


def cholesky(a, *, mode: str = "ldlt", ordering: str = "amd", lower: bool = True, shift: float = 0.0) -> SparseCholesky:
    """Factorize a sparse symmetric matrix; see :class:`SparseCholesky`."""
    return SparseCholesky(a, mode=mode, ordering=ordering, lower=lower, shift=shift)


def to_dense(a):
    """Convert sparse matrix to dense ndarray."""
    return np.asarray(a.toarray(), dtype=np.float64)


def from_coo(data, row, col, shape):
    """Build sparse matrix from COO inputs."""
    sp = _require_scipy()
    return sp.coo_matrix((data, (row, col)), shape=shape).tocsc()
