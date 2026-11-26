"""Tests specific to mode="multifrontal" (Eigen MultifrontalLDLT): indefinite input, 2x2 pivots,
inertia, and the shift applied through an explicit diagonal. Generic SPD behaviour is covered by
test_sparse_cholesky.py, which runs every mode."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numpy.testing as npt
import pytest

sp = pytest.importorskip("scipy.sparse")

from peigen import sparse

ORDERINGS = ["amd", "natural"]


def laplacian_2d(nx):
    t = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(nx, nx))
    return (sp.kron(sp.eye(nx), t) + sp.kron(t, sp.eye(nx))).tocsc()


def laplacian_3d(nx):
    t = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(nx, nx))
    eye = sp.eye(nx)
    return (sp.kron(sp.kron(t, eye), eye) + sp.kron(sp.kron(eye, t), eye) + sp.kron(sp.kron(eye, eye), t)).tocsc()


def kkt(n=40, m=11, seed=0):
    """Saddle-point matrix [[H, B^T], [B, 0]] with H SPD and B full row rank: inertia (n, m, 0)."""
    rng = np.random.default_rng(seed)
    base = sp.random(n, n, density=0.1, format="csc", random_state=seed)
    h = (base.T @ base + 2.0 * sp.eye(n)).tocsc()
    b = sp.random(m, n, density=0.3, format="csc", random_state=seed + 1) + sp.eye(m, n)
    b = sp.csc_matrix(b.toarray() + 0.01 * rng.standard_normal((m, n)) * (b.toarray() != 0))
    return sp.bmat([[h, b.T], [b, None]], format="csc"), n, m


@pytest.mark.parametrize("ordering", ORDERINGS)
def test_kkt_solve_inertia_slogdet(ordering):
    a, n, m = kkt()
    dense = a.toarray()
    fac = sparse.cholesky(a, mode="multifrontal", ordering=ordering)
    rng = np.random.default_rng(3)
    b = rng.standard_normal((n + m, 3))
    npt.assert_allclose(fac.solve(b), np.linalg.solve(dense, b), rtol=1e-9, atol=1e-10)
    assert fac.inertia == (n, m, 0)
    sign, logabs = fac.slogdet()
    ref_sign, ref_logabs = np.linalg.slogdet(dense)
    assert sign == ref_sign == (-1.0) ** m
    npt.assert_allclose(logabs, ref_logabs, rtol=1e-10)
    with pytest.raises(ValueError, match="not positive"):
        fac.logdet()


def test_zero_diagonal_needs_2x2_pivots():
    # Block diagonal of [[0, 1], [1, 0]] blocks: every diagonal entry is zero, so any 1x1 pivot
    # strategy breaks down, but 2x2 pivots factor it exactly.
    k = 25
    a = sp.block_diag([sp.csc_matrix([[0.0, 1.0], [1.0, 0.0]])] * k, format="csc")
    with pytest.raises(np.linalg.LinAlgError):
        sparse.cholesky(a, mode="ldlt")
    fac = sparse.cholesky(a, mode="multifrontal")
    b = np.arange(2 * k, dtype=float)
    npt.assert_allclose(a @ fac.solve(b), b, atol=1e-12)
    assert fac.inertia == (k, k, 0)
    assert fac.slogdet() == ((-1.0) ** k, 0.0)
    bb = np.column_stack([b, -2.0 * b, np.ones(2 * k)])  # several RHS: blocked path, 2x2 D blocks
    npt.assert_allclose(a @ fac.solve(bb), bb, atol=1e-12)


def test_indefinite_more_accurate_than_simplicial_ldlt():
    # Tiny diagonal entries next to large off-diagonals: unpivoted LDLT amplifies rounding error.
    rng = np.random.default_rng(7)
    n = 60
    off = sp.random(n, n, density=0.15, format="csc", random_state=7)
    a = (off + off.T + sp.diags(1e-8 * rng.standard_normal(n))).tocsc()
    a = (a + sp.diags(np.where(np.arange(n) % 2, 1.0, -1.0) * 1e-8)).tocsc()
    dense = a.toarray()
    if abs(np.linalg.det(dense)) < 1e-30 or np.linalg.cond(dense) > 1e10:
        pytest.skip("random matrix too ill-conditioned for a meaningful comparison")
    b = rng.standard_normal(n)
    mf = np.linalg.norm(dense @ sparse.cholesky(a, mode="multifrontal").solve(b) - b)
    assert mf < 1e-9 * np.linalg.norm(b) * np.linalg.cond(dense)
    try:
        ld = np.linalg.norm(dense @ sparse.cholesky(a, mode="ldlt").solve(b) - b)
    except np.linalg.LinAlgError:
        return
    assert mf <= ld


def test_shift_on_matrix_without_stored_diagonal():
    # Off-diagonal-only pattern: the shift has to add diagonal entries the input does not store.
    n = 30
    a = sp.diags([1.0, 1.0], [-1, 1], shape=(n, n), format="csc")
    assert a.diagonal().tolist() == [0.0] * n and a.nnz == 2 * (n - 1)
    fac = sparse.cholesky(a, mode="multifrontal", shift=3.0)
    b = np.ones(n)
    npt.assert_allclose(fac.solve(b), np.linalg.solve(a.toarray() + 3.0 * np.eye(n), b), rtol=1e-12)
    assert fac.shift == 3.0 and fac.inertia == (n, 0, 0)

    # Same pattern, new shift and new values: reuses the analysis.
    fac.refactorize(2.0 * a, shift=-5.0)
    npt.assert_allclose(fac.solve(b), np.linalg.solve(2.0 * a.toarray() - 5.0 * np.eye(n), b), rtol=1e-12)
    assert fac.inertia == (0, n, 0)


def test_refactorize_changes_inertia():
    a = laplacian_2d(8)
    fac = sparse.cholesky(a, mode="multifrontal")
    assert fac.inertia == (64, 0, 0)
    eig = np.linalg.eigvalsh(a.toarray())
    shift = -0.5 * (eig[10] + eig[11])  # puts exactly 11 eigenvalues below zero
    fac.refactorize(a, shift=shift)
    assert fac.inertia == (53, 11, 0)
    npt.assert_allclose(fac.slogdet()[1], np.linalg.slogdet(a.toarray() + shift * np.eye(64))[1], rtol=1e-10)


@pytest.mark.parametrize("ordering", ORDERINGS)
@pytest.mark.parametrize(
    "make",
    [lambda: laplacian_3d(12), lambda: laplacian_3d(12) - 1.5 * sp.eye(12**3), lambda: kkt(300, 80)[0]],
    ids=["3d-spd", "3d-indefinite", "kkt"],
)
def test_many_rhs_matches_single_rhs(make, ordering):
    # Many RHS go through the supernodal blocked solve (dense kernels on large fronts, row updates
    # on small ones); one RHS goes through Eigen's solve. Both must agree with a dense solve.
    a = make()
    fac = sparse.cholesky(a, mode="multifrontal", ordering=ordering)
    rng = np.random.default_rng(5)
    b = rng.standard_normal((a.shape[0], 9))
    ref = np.linalg.solve(a.toarray(), b)
    x = fac.solve(b)
    npt.assert_allclose(x, ref, rtol=1e-8, atol=1e-10 * np.abs(ref).max())
    npt.assert_allclose(np.column_stack([fac.solve(b[:, j]) for j in range(9)]), x, rtol=1e-8, atol=1e-10)
    npt.assert_allclose(fac.solve(b[:, :2]), x[:, :2], rtol=1e-8, atol=1e-10)


def test_refactorize_invalidates_blocked_solve():
    a = laplacian_3d(8)
    fac = sparse.cholesky(a, mode="multifrontal")
    b = np.random.default_rng(1).standard_normal((a.shape[0], 4))
    fac.solve(b)  # builds the blocked factor
    fac.refactorize(a, shift=0.7)
    npt.assert_allclose(fac.solve(b), np.linalg.solve(a.toarray() + 0.7 * np.eye(a.shape[0]), b), rtol=1e-10)


def test_inertia_simplicial_modes():
    a = laplacian_2d(5)
    assert sparse.cholesky(a, mode="llt").inertia == (25, 0, 0)
    assert sparse.cholesky(a, mode="ldlt").inertia == (25, 0, 0)
    assert sparse.cholesky(-a, mode="ldlt").inertia == (0, 25, 0)


def test_singular_and_nonfinite_raise():
    with pytest.raises(np.linalg.LinAlgError, match="multifrontal"):
        sparse.cholesky(sp.csc_matrix(np.diag([1.0, 0.0, 2.0])), mode="multifrontal")
    bad = laplacian_2d(4).tolil()
    bad[3, 3] = np.nan
    with pytest.raises(np.linalg.LinAlgError):
        sparse.cholesky(bad.tocsc(), mode="multifrontal")


def test_upper_triangle_only():
    a, n, m = kkt(20, 6, seed=4)
    b = np.linspace(-1.0, 1.0, n + m)
    ref = np.linalg.solve(a.toarray(), b)
    npt.assert_allclose(sparse.cholesky(sp.triu(a), mode="multifrontal", lower=False).solve(b), ref, rtol=1e-9)
    npt.assert_allclose(sparse.cholesky(sp.tril(a), mode="multifrontal").solve(b), ref, rtol=1e-9)


def test_concurrent_solves_share_one_factorization():
    a = laplacian_2d(40)
    fac = sparse.cholesky(a, mode="multifrontal")
    rng = np.random.default_rng(0)
    rhs = [rng.standard_normal((a.shape[0], k)) for k in (1, 4, 1, 16, 1, 4, 1, 16)]
    expected = [fac.solve(b) for b in rhs]
    # A fresh factorization, so the threads also race to build the blocked factor.
    fac = sparse.cholesky(a, mode="multifrontal")
    with ThreadPoolExecutor(max_workers=4) as pool:
        got = list(pool.map(fac.solve, rhs))
    for e, g in zip(expected, got):
        npt.assert_array_equal(e, g)


def test_invalid_mode_lists_multifrontal():
    with pytest.raises(ValueError, match="multifrontal"):
        sparse.cholesky(laplacian_2d(3), mode="supernodal")
