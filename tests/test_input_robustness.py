"""Regression tests for input handling: index/data dtypes, non-canonical sparse, complex, scaling."""

import numpy as np
import numpy.testing as npt
import pytest

from peigen import decomp, linalg, sparse

sp = pytest.importorskip("scipy.sparse")


def _spd(n=60, seed=0):
    base = sp.random(n, n, density=0.08, format="csc", random_state=seed)
    return ((base.T @ base) + 5.0 * sp.eye(n, format="csc")).tocsc()


def _general(n=60, seed=1):
    return (sp.random(n, n, density=0.08, format="csc", random_state=seed) + 6.0 * sp.eye(n, format="csc")).tocsc()


def _with_int64_indices(a):
    out = a.copy()
    out.indptr = out.indptr.astype(np.int64)
    out.indices = out.indices.astype(np.int64)
    assert out.indices.dtype == np.int64
    return out


SPARSE_VARIANTS = {
    "int64_indices": _with_int64_indices,
    "float32_data": lambda a: a.astype(np.float32),
    "int64_and_float32": lambda a: _with_int64_indices(a.astype(np.float32)),
    "csr": lambda a: a.tocsr(),
    "csc_array": lambda a: sp.csc_array(a),
    "coo": lambda a: a.tocoo(),
}


@pytest.fixture(params=sorted(SPARSE_VARIANTS))
def variant(request):
    return SPARSE_VARIANTS[request.param]


# --- dtype / format variants: previously crashed (int64) or returned garbage (float32) ---------


def test_spmm_variants(variant):
    a = _general()
    av = variant(a)
    ref = sp.csc_matrix(av, dtype=np.float64)
    b = np.random.default_rng(0).standard_normal((a.shape[1], 3))
    for _ in range(5):  # repeat: dangling-pointer reads were nondeterministic
        npt.assert_allclose(sparse.spmm(av, b), ref @ b, rtol=1e-12, atol=1e-12)


def test_spspmm_variants(variant):
    a, b = _general(seed=2), _general(seed=3)
    av, bv = variant(a), variant(b)
    ref = sp.csc_matrix(av, dtype=np.float64) @ sp.csc_matrix(bv, dtype=np.float64)
    npt.assert_allclose(sparse.spspmm(av, bv).toarray(), ref.toarray(), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("method", ["lu", "bicgstab"])
def test_sparse_solve_variants(variant, method):
    a = _general()
    av = variant(a)
    ref = sp.csc_matrix(av, dtype=np.float64)
    b = np.random.default_rng(1).standard_normal(a.shape[0])
    x = sparse.solve(av, b, method=method, tol=1e-12)
    assert np.linalg.norm(ref @ x - b) / np.linalg.norm(b) < 1e-8


def test_sparse_cg_and_stats_variants(variant):
    a = _spd()
    av = variant(a)
    ref = sp.csc_matrix(av, dtype=np.float64)
    b = np.random.default_rng(2).standard_normal(a.shape[0])
    x = sparse.solve(av, b, method="cg", preconditioner="jacobi", tol=1e-12, maxiter=5000)
    assert np.linalg.norm(ref @ x - b) / np.linalg.norm(b) < 1e-8
    stats = sparse.solve_stats(av, b, method="cg", preconditioner="ilu", ilu_fill_factor=40)
    assert stats["iterations"] > 0


def test_factorize_variants(variant):
    a = _general()
    av = variant(a)
    ref = sp.csc_matrix(av, dtype=np.float64)
    fac = sparse.factorize(av)
    b = np.random.default_rng(3).standard_normal((a.shape[0], 2))
    x = fac.solve(b)
    assert np.linalg.norm(ref @ x - b) / np.linalg.norm(b) < 1e-10


# --- non-canonical CSC ------------------------------------------------------------------------


def _non_canonical():
    # Column 0 has unsorted rows (2, 0) and a duplicate entry at row 2; column 1 is unsorted too.
    data = np.array([4.0, 1.0, 3.0, 2.0, 5.0, 7.0])
    indices = np.array([2, 0, 2, 1, 0, 2])
    indptr = np.array([0, 3, 5, 6])
    a = sp.csc_matrix((data, indices, indptr), shape=(3, 3))
    assert not a.has_canonical_format
    return a


def test_non_canonical_sparse_is_handled_and_not_mutated():
    a = _non_canonical()
    snapshot = (a.data.copy(), a.indices.copy(), a.indptr.copy())
    dense = a.toarray()
    b = np.array([1.0, 2.0, 3.0])

    npt.assert_allclose(sparse.spmm(a, b[:, None])[:, 0], dense @ b)
    npt.assert_allclose(sparse.spspmm(a, a).toarray(), dense @ dense)
    npt.assert_allclose(sparse.solve(a, b, method="lu"), np.linalg.solve(dense, b), rtol=1e-12)
    npt.assert_allclose(sparse.factorize(a).solve(b), np.linalg.solve(dense, b), rtol=1e-12)

    npt.assert_array_equal(a.data, snapshot[0])
    npt.assert_array_equal(a.indices, snapshot[1])
    npt.assert_array_equal(a.indptr, snapshot[2])


# --- factorize() API --------------------------------------------------------------------------


def test_factorize_solve_accepts_1d_rhs():
    a = _general()
    b = np.random.default_rng(4).standard_normal(a.shape[0])
    fac = sparse.factorize(a)
    x = fac.solve(b)
    assert x.shape == b.shape
    npt.assert_allclose(a @ x, b, rtol=1e-10, atol=1e-10)
    assert fac.shape == a.shape


def test_factorize_is_decomp_sparse_factorized():
    a = _general()
    assert isinstance(sparse.factorize(a), decomp.SparseFactorized)


def test_factorize_shape_mismatch():
    fac = sparse.factorize(_general(n=10))
    with pytest.raises(ValueError):
        fac.solve(np.ones(11))


def test_factorize_singular_raises():
    a = sp.csc_matrix(np.array([[1.0, 2.0], [2.0, 4.0]]))
    with pytest.raises(RuntimeError):
        sparse.factorize(a)


# --- complex input is rejected instead of silently truncated ----------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda z: linalg.matmul(z, z),
        lambda z: linalg.solve(z, np.ones(3)),
        lambda z: linalg.solve(np.eye(3), z[:, 0]),
        lambda z: linalg.qr(z),
        lambda z: linalg.svd(z),
        lambda z: linalg.eigh(z),
        lambda z: linalg.eighvals(z),
        lambda z: decomp.SVD(z),
        lambda z: decomp.SVD(np.eye(3)).solve(z[:, 0]),
    ],
)
def test_dense_complex_rejected(call):
    z = np.eye(3) + 1j * np.ones((3, 3))
    with pytest.raises(TypeError, match="complex"):
        call(z)


@pytest.mark.parametrize(
    "call",
    [
        lambda z, b: sparse.spmm(z, b[:, None]),
        lambda z, b: sparse.spspmm(z, z),
        lambda z, b: sparse.solve(z, b),
        lambda z, b: sparse.factorize(z),
        lambda z, b: sparse.solve(sp.eye(3, format="csc"), b + 1j),
    ],
)
def test_sparse_complex_rejected(call):
    z = sp.csc_matrix(np.eye(3) + 1j * np.eye(3))
    with pytest.raises(TypeError, match="complex"):
        call(z, np.ones(3))


def test_norm_complex_matches_numpy():
    z = np.arange(6.0).reshape(2, 3) + 1j
    assert np.isclose(linalg.norm(z), np.linalg.norm(z))


# --- scale-aware tolerances ---------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1e-16, 1e-200, 1e200])
def test_eigen_solve_accepts_well_conditioned_scaled_matrix(scale):
    a = scale * (np.eye(4) + 0.1 * np.ones((4, 4)))
    b = np.ones(4)
    npt.assert_allclose(linalg.solve(a, b, method="eigen"), np.linalg.solve(a, b), rtol=1e-12)


@pytest.mark.parametrize("method", ["eigen", "auto"])
def test_solve_singular_still_raises(method):
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])  # rank 2, rounding-level pivot
    with pytest.raises(ValueError):
        linalg.solve(np.zeros((3, 3)), np.ones(3), method=method)
    if method == "eigen":
        with pytest.raises(ValueError):
            linalg.solve(a, np.ones(3), method=method)


@pytest.mark.parametrize("scale", [1e-14, 1.0, 1e14])
def test_svd_solve_is_scale_invariant(scale):
    rng = np.random.default_rng(5)
    a = scale * rng.standard_normal((20, 5))
    b = rng.standard_normal(20)
    x_ref, *_ = np.linalg.lstsq(a, b, rcond=None)
    npt.assert_allclose(decomp.SVD(a).solve(b), x_ref, rtol=1e-9)


def test_svd_solve_rank_deficient_matches_lstsq():
    rng = np.random.default_rng(6)
    a = rng.standard_normal((15, 3)) @ rng.standard_normal((3, 8))  # rank 3
    b = rng.standard_normal((15, 2))
    x_ref, *_ = np.linalg.lstsq(a, b, rcond=None)
    npt.assert_allclose(decomp.SVD(a).solve(b), x_ref, rtol=1e-8, atol=1e-10)


def test_svd_solve_full_matrices():
    rng = np.random.default_rng(7)
    for shape in [(12, 5), (5, 12)]:
        a = rng.standard_normal(shape)
        b = rng.standard_normal(shape[0])
        x_ref, *_ = np.linalg.lstsq(a, b, rcond=None)
        npt.assert_allclose(decomp.SVD(a, full_matrices=True).solve(b), x_ref, rtol=1e-9, atol=1e-12)
