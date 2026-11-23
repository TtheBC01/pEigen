import numpy as np
import numpy.testing as npt
import pytest

sp = pytest.importorskip("scipy.sparse")

from peigen import decomp, sparse

MODES = ["ldlt", "llt"]
ORDERINGS = ["amd", "natural"]


def laplacian_2d(nx, ny=None):
    """5-point Dirichlet Laplacian on an nx-by-ny grid (SPD)."""
    ny = nx if ny is None else ny
    tx = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(nx, nx))
    ty = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(ny, ny))
    return (sp.kron(sp.eye(ny), tx) + sp.kron(ty, sp.eye(nx))).tocsc()


def random_spd(n=50, seed=0):
    base = sp.random(n, n, density=0.08, format="csc", random_state=seed)
    return ((base.T @ base) + 3.0 * sp.eye(n)).tocsc()


@pytest.mark.parametrize("ordering", ORDERINGS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("make", [lambda: laplacian_2d(12, 9), random_spd], ids=["laplacian", "random_spd"])
def test_solve_matches_dense(mode, ordering, make):
    a = make()
    rng = np.random.default_rng(0)
    dense = a.toarray()
    fac = sparse.cholesky(a, mode=mode, ordering=ordering)
    assert fac.shape == a.shape and fac.mode == mode and fac.ordering == ordering

    b1 = rng.standard_normal(a.shape[0])
    x1 = fac.solve(b1)
    assert x1.shape == b1.shape
    npt.assert_allclose(x1, np.linalg.solve(dense, b1), rtol=1e-10, atol=1e-12)

    b2 = rng.standard_normal((a.shape[0], 4))
    npt.assert_allclose(fac.solve(b2), np.linalg.solve(dense, b2), rtol=1e-10, atol=1e-12)
    # non-contiguous rhs
    npt.assert_allclose(fac.solve(b2[:, ::2]), np.linalg.solve(dense, b2[:, ::2]), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("ordering", ORDERINGS)
@pytest.mark.parametrize("mode", MODES)
def test_many_rhs_blocked_solve(mode, ordering):
    a = random_spd(80, seed=11)
    rng = np.random.default_rng(1)
    for k in (1, 2, 33):
        b = rng.standard_normal((a.shape[0], k))
        fac = sparse.cholesky(a, mode=mode, ordering=ordering, shift=0.25)
        ref = np.linalg.solve(a.toarray() + 0.25 * np.eye(a.shape[0]), b)
        npt.assert_allclose(fac.solve(b), ref, rtol=1e-10, atol=1e-12)
        npt.assert_allclose(fac.solve(np.asfortranarray(b)), ref, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("mode", MODES)
def test_only_one_triangle_is_read(mode):
    a = random_spd(seed=1)
    b = np.ones(a.shape[0])
    ref = np.linalg.solve(a.toarray(), b)
    npt.assert_allclose(sparse.cholesky(sp.tril(a).tocsc(), mode=mode).solve(b), ref, rtol=1e-10)
    npt.assert_allclose(sparse.cholesky(sp.triu(a).tocsc(), mode=mode, lower=False).solve(b), ref, rtol=1e-10)
    # garbage in the unread triangle is ignored
    junk = (sp.tril(a) + sp.triu(sp.random(*a.shape, density=0.2, random_state=3), k=1)).tocsc()
    npt.assert_allclose(sparse.cholesky(junk, mode=mode).solve(b), ref, rtol=1e-10)


@pytest.mark.parametrize("mode", MODES)
def test_shift(mode):
    a = laplacian_2d(8)
    b = np.arange(a.shape[0], dtype=float)
    fac = sparse.cholesky(a, mode=mode, shift=0.5)
    assert fac.shift == 0.5
    npt.assert_allclose(fac.solve(b), np.linalg.solve(a.toarray() + 0.5 * np.eye(a.shape[0]), b), rtol=1e-10)


@pytest.mark.parametrize("mode", MODES)
def test_refactorize_same_pattern(mode):
    a = laplacian_2d(10)
    n = a.shape[0]
    b = np.ones(n)
    fac = sparse.cholesky(a, mode=mode)

    a2 = a.copy()
    a2.data = a2.data * 3.0
    a2.setdiag(a2.diagonal() + np.linspace(1.0, 2.0, n))
    assert fac.refactorize(a2) is fac
    npt.assert_allclose(fac.solve(b), np.linalg.solve(a2.toarray(), b), rtol=1e-10)

    fac.refactorize(a2, shift=2.0)
    npt.assert_allclose(fac.solve(b), np.linalg.solve(a2.toarray() + 2.0 * np.eye(n), b), rtol=1e-10)
    fac.refactorize(a2)  # shift=None keeps the current shift
    assert fac.shift == 2.0


def test_refactorize_rejects_different_pattern():
    a = laplacian_2d(6)
    fac = sparse.cholesky(a)
    other = (a + sp.random(*a.shape, density=0.05, random_state=4) * 0.01).tocsc()
    other = ((other + other.T) / 2).tocsc()
    with pytest.raises(ValueError, match="sparsity pattern"):
        fac.refactorize(other)
    with pytest.raises(ValueError, match="sparsity pattern"):
        fac.refactorize(laplacian_2d(7))
    # factorization remains usable after a rejected refactorize
    npt.assert_allclose(a @ fac.solve(np.ones(a.shape[0])), np.ones(a.shape[0]), rtol=1e-10)


@pytest.mark.parametrize("mode", MODES)
def test_slogdet_matches_numpy(mode):
    for a in (laplacian_2d(7), random_spd(30, seed=5), 1e-3 * random_spd(30, seed=6)):
        sign, logabs = sparse.cholesky(a, mode=mode).slogdet()
        ref_sign, ref_logabs = np.linalg.slogdet(a.toarray())
        assert sign == ref_sign == 1.0
        assert np.isclose(logabs, ref_logabs, rtol=1e-11)
        assert np.isclose(sparse.cholesky(a, mode=mode).logdet(), ref_logabs, rtol=1e-11)


def test_slogdet_large_does_not_overflow():
    a = (1e3 * laplacian_2d(30)).tocsc()  # det ~ 1e2700, overflows float64
    _, logabs = sparse.cholesky(a).slogdet()
    assert np.isfinite(logabs) and logabs > 700


def test_ldlt_indefinite_sign_and_llt_rejects():
    a = sp.csc_matrix(np.diag([2.0, -3.0, 4.0]) + np.diag([0.5, 0.5], 1) + np.diag([0.5, 0.5], -1))
    ldlt = sparse.cholesky(a, mode="ldlt")
    sign, logabs = ldlt.slogdet()
    ref_sign, ref_logabs = np.linalg.slogdet(a.toarray())
    assert sign == ref_sign == -1.0
    assert np.isclose(logabs, ref_logabs)
    with pytest.raises(ValueError):
        ldlt.logdet()
    npt.assert_allclose(ldlt.solve(np.ones(3)), np.linalg.solve(a.toarray(), np.ones(3)), rtol=1e-12)

    with pytest.raises(np.linalg.LinAlgError, match="positive definite"):
        sparse.cholesky(a, mode="llt")


def test_singular_raises_linalg_error():
    a = sp.csc_matrix(np.array([[1.0, 1.0], [1.0, 1.0]]))
    with pytest.raises(np.linalg.LinAlgError):
        sparse.cholesky(a, mode="ldlt")
    with pytest.raises(np.linalg.LinAlgError):
        sparse.cholesky(a, mode="llt")


def test_failed_refactorize_raises():
    a = laplacian_2d(5)
    fac = sparse.cholesky(a, mode="llt")
    with pytest.raises(np.linalg.LinAlgError):
        fac.refactorize(-a)


def test_amd_reduces_fill():
    a = laplacian_2d(40)
    assert sparse.cholesky(a, ordering="amd").factor_nnz < sparse.cholesky(a, ordering="natural").factor_nnz


@pytest.mark.parametrize(
    "a_variant",
    [
        lambda a: a.tocsr(),
        lambda a: sp.csc_array(a),
        lambda a: a.astype(np.float32),
        lambda a: a.toarray(),
    ],
    ids=["csr", "csc_array", "float32", "dense"],
)
def test_input_variants(a_variant):
    a = random_spd(seed=7)
    ref = sp.csc_matrix(a_variant(a), dtype=np.float64).toarray()
    b = np.ones(a.shape[0])
    npt.assert_allclose(sparse.cholesky(a_variant(a)).solve(b), np.linalg.solve(ref, b), rtol=1e-6)


def test_int64_indices():
    a = random_spd(seed=8)
    a64 = a.copy()
    a64.indptr = a64.indptr.astype(np.int64)
    a64.indices = a64.indices.astype(np.int64)
    fac = sparse.cholesky(a64)
    fac.refactorize(a64)
    b = np.ones(a.shape[0])
    npt.assert_allclose(fac.solve(b), np.linalg.solve(a.toarray(), b), rtol=1e-10)


def test_errors():
    with pytest.raises(ValueError, match="square"):
        sparse.cholesky(sp.random(4, 5, density=0.5, format="csc"))
    with pytest.raises(ValueError, match="mode"):
        sparse.cholesky(sp.eye(3, format="csc"), mode="qr")
    with pytest.raises(ValueError, match="ordering"):
        sparse.cholesky(sp.eye(3, format="csc"), ordering="metis")
    fac = sparse.cholesky(sp.eye(3, format="csc"))
    with pytest.raises(ValueError):
        fac.solve(np.ones(4))
    with pytest.raises(TypeError, match="complex"):
        fac.solve(np.ones(3) * 1j)
    with pytest.raises(TypeError, match="complex"):
        sparse.cholesky(sp.eye(3, format="csc") * 1j)


def test_sparse_solve_cholesky_method():
    a = laplacian_2d(9)
    b = np.random.default_rng(9).standard_normal((a.shape[0], 2))
    npt.assert_allclose(sparse.solve(a, b, method="cholesky"), np.linalg.solve(a.toarray(), b), rtol=1e-10)
    npt.assert_allclose(sparse.solve(a, b[:, 0], method="cholesky"), np.linalg.solve(a.toarray(), b[:, 0]), rtol=1e-10)
    with pytest.raises(ValueError):
        sparse.solve(a, b, method="cholesky", preconditioner="jacobi")


def test_decomp_alias():
    assert decomp.SparseCholesky is sparse.SparseCholesky
    assert isinstance(sparse.cholesky(sp.eye(2, format="csc")), decomp.SparseCholesky)
