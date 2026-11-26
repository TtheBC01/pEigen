"""SparseCholesky.L(), D() and P(): the factors themselves, for every mode."""

import numpy as np
import numpy.testing as npt
import pytest

sp = pytest.importorskip("scipy.sparse")
spla = pytest.importorskip("scipy.sparse.linalg")

from peigen import sparse

MODES = ["ldlt", "llt", "multifrontal"]
ORDERINGS = ["amd", "natural"]


def laplacian_2d(m):
    t = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(m, m))
    return (sp.kron(sp.eye(m), t) + sp.kron(t, sp.eye(m))).tocsc()


def kkt(n=30, m=8, seed=0):
    base = sp.random(n, n, density=0.15, format="csc", random_state=seed)
    h = (base.T @ base + 2.0 * sp.eye(n)).tocsc()
    b = sp.random(m, n, density=0.3, format="csc", random_state=seed + 1) + sp.eye(m, n)
    return sp.bmat([[h, b.T], [b, None]], format="csc")


def reconstruct(fac):
    L, D, p = fac.L(), fac.D(), fac.P()
    return p, (L @ D @ L.T).toarray()


@pytest.mark.parametrize("ordering", ORDERINGS)
@pytest.mark.parametrize("mode", MODES)
def test_factors_reconstruct_shifted_matrix(mode, ordering):
    a = laplacian_2d(9)
    fac = sparse.cholesky(a, mode=mode, ordering=ordering, shift=0.25)
    p, ldlt = reconstruct(fac)
    shifted = a.toarray() + 0.25 * np.eye(a.shape[0])
    npt.assert_allclose(ldlt, shifted[np.ix_(p, p)], atol=1e-12)

    L = fac.L()
    assert sp.issparse(L) and L.format == "csc" and L.shape == a.shape
    assert sorted(p.tolist()) == list(range(a.shape[0]))
    npt.assert_array_equal(sp.triu(L, 1).toarray(), 0.0)
    diag = L.diagonal()
    if mode == "llt":
        assert np.all(diag > 0)
        npt.assert_array_equal(fac.D().toarray(), np.eye(a.shape[0]))
        assert L.nnz == fac.factor_nnz
    else:
        npt.assert_array_equal(diag, 1.0)
        assert L.nnz == fac.factor_nnz + a.shape[0]


def test_natural_ordering_is_identity_for_simplicial():
    a = laplacian_2d(6)
    for mode in ("ldlt", "llt"):
        npt.assert_array_equal(sparse.cholesky(a, mode=mode, ordering="natural").P(), np.arange(36))


def test_upper_triangle_input():
    a = laplacian_2d(7)
    fac = sparse.cholesky(sp.triu(a).tocsc(), lower=False)
    p, ldlt = reconstruct(fac)
    npt.assert_allclose(ldlt, a.toarray()[np.ix_(p, p)], atol=1e-12)


@pytest.mark.parametrize("ordering", ORDERINGS)
def test_multifrontal_2x2_blocks_in_D(ordering):
    # Every diagonal entry is zero, so every pivot has to be a 2x2 block.
    k = 10
    a = sp.block_diag([sp.csc_matrix([[0.0, 2.0], [2.0, 0.0]])] * k, format="csc")
    fac = sparse.cholesky(a, mode="multifrontal", ordering=ordering)
    D = fac.D().toarray()
    npt.assert_array_equal(D, D.T)
    assert np.count_nonzero(np.diag(D, -1)) == k  # one off-diagonal per 2x2 block
    p, ldlt = reconstruct(fac)
    npt.assert_allclose(ldlt, a.toarray()[np.ix_(p, p)], atol=1e-12)
    eig = np.linalg.eigvalsh(D)
    assert ((eig > 0).sum(), (eig < 0).sum(), 0) == fac.inertia


def test_multifrontal_kkt_reconstructs_and_inertia_matches_D():
    a = kkt()
    fac = sparse.cholesky(a, mode="multifrontal")
    p, ldlt = reconstruct(fac)
    npt.assert_allclose(ldlt, a.toarray()[np.ix_(p, p)], atol=1e-10)
    eig = np.linalg.eigvalsh(fac.D().toarray())
    assert ((eig > 0).sum(), (eig < 0).sum(), 0) == fac.inertia == (30, 8, 0)


@pytest.mark.parametrize("mode", MODES)
def test_factors_follow_refactorize(mode):
    a = laplacian_2d(8)
    fac = sparse.cholesky(a, mode=mode)
    fac.refactorize(3.0 * a, shift=1.5)
    p, ldlt = reconstruct(fac)
    npt.assert_allclose(ldlt, (3.0 * a.toarray() + 1.5 * np.eye(64))[np.ix_(p, p)], atol=1e-11)


@pytest.mark.parametrize("mode", MODES)
def test_sampling_recipe_has_covariance_inverse(mode):
    # x[p] = y with L^T y = D^{-1/2} z gives x ~ N(0, As^{-1}). Applying the map to the identity
    # gives a matrix S with S S^T = As^{-1}.
    a = laplacian_2d(6)
    fac = sparse.cholesky(a, mode=mode, shift=0.5)
    L, p = fac.L(), fac.P()
    d = fac.D().diagonal()
    n = a.shape[0]
    y = spla.spsolve_triangular(L.T.tocsr(), np.diag(1.0 / np.sqrt(d)), lower=False)
    S = np.empty_like(y)
    S[p] = y
    cov = np.linalg.inv(a.toarray() + 0.5 * np.eye(n))
    npt.assert_allclose(S @ S.T, cov, atol=1e-12)
