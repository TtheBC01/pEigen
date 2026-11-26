import numpy as np
import numpy.testing as npt
import pytest

sp = pytest.importorskip("scipy.sparse")

from peigen import sparse


@pytest.mark.sparse
def test_spmm_matches_scipy():
    rng = np.random.default_rng(10)
    a = sp.random(60, 40, density=0.07, format="csc", random_state=10)
    b = rng.standard_normal((40, 15))
    npt.assert_allclose(sparse.spmm(a, b), a @ b, rtol=1e-10, atol=1e-10)


@pytest.mark.sparse
def test_spspmm_matches_scipy():
    a = sp.random(50, 35, density=0.05, format="csc", random_state=11)
    b = sp.random(35, 20, density=0.08, format="csc", random_state=12)
    out = sparse.spspmm(a, b)
    npt.assert_allclose(out.toarray(), (a @ b).toarray(), rtol=1e-10, atol=1e-10)


@pytest.mark.sparse
def test_sparse_solve_matches_scipy_lu():
    rng = np.random.default_rng(13)
    a = sp.random(30, 30, density=0.08, format="csc", random_state=13)
    a = a + 10.0 * sp.eye(30, format="csc")
    b = rng.standard_normal((30, 2))

    x = sparse.solve(a, b, method="lu")
    resid = np.linalg.norm(a @ x - b) / np.linalg.norm(b)
    assert resid < 1e-9


@pytest.mark.sparse
def test_sparse_solve_cg_respects_tol_maxiter():
    rng = np.random.default_rng(15)
    base = sp.random(40, 40, density=0.06, format="csc", random_state=15)
    a = (base.T @ base) + 5.0 * sp.eye(40, format="csc")
    b = rng.standard_normal((40, 1))

    x = sparse.solve(a, b, method="cg", tol=1e-10, maxiter=4000)
    resid = np.linalg.norm(a @ x - b) / np.linalg.norm(b)
    assert resid < 1e-7


@pytest.mark.parametrize("preconditioner", ["none", "jacobi", "ichol"])
@pytest.mark.sparse
def test_sparse_solve_cg_preconditioners(preconditioner):
    rng = np.random.default_rng(17)
    base = sp.random(40, 40, density=0.06, format="csc", random_state=17)
    a = (base.T @ base) + 5.0 * sp.eye(40, format="csc")
    b = rng.standard_normal(40)

    solve_kwargs = {
        "method": "cg",
        "tol": 1e-8,
        "maxiter": 4000,
        "preconditioner": preconditioner,
    }

    x = sparse.solve(a, b, **solve_kwargs)
    resid = np.linalg.norm(a @ x - b) / np.linalg.norm(b)
    assert resid < 1e-7

    stats = sparse.solve_stats(a, b, **solve_kwargs)
    assert stats["iterations"] >= 0
    assert stats["error"] >= 0.0
    npt.assert_allclose(stats["residual"], resid, rtol=1e-6, atol=1e-15)
    assert stats["residual"] < 1e-7


def _laplacian_2d(m):
    t = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(m, m))
    eye = sp.eye(m)
    return (sp.kron(eye, t) + sp.kron(t, eye)).tocsc()


@pytest.mark.parametrize("m", [30, 60, 100])
@pytest.mark.sparse
def test_sparse_solve_cg_ichol_laplacian_default_settings(m):
    # Regression: CG + ILU (IncompleteLUT, not symmetric) failed to converge here with default
    # settings. Incomplete Cholesky must converge, and in fewer iterations than plain CG.
    a = _laplacian_2d(m)
    b = np.ones(m * m)

    x = sparse.solve(a, b, method="cg", preconditioner="ichol")
    assert np.linalg.norm(a @ x - b) / np.linalg.norm(b) < 1e-7

    plain = sparse.solve_stats(a, b, method="cg")
    ichol = sparse.solve_stats(a, b, method="cg", preconditioner="ichol")
    assert ichol["residual"] < 1e-7
    assert ichol["iterations"] < plain["iterations"]


@pytest.mark.sparse
def test_sparse_solve_cg_ilu_is_deprecated():
    rng = np.random.default_rng(17)
    base = sp.random(40, 40, density=0.06, format="csc", random_state=17)
    a = (base.T @ base) + 5.0 * sp.eye(40, format="csc")
    b = rng.standard_normal(40)
    kwargs = {"method": "cg", "preconditioner": "ilu", "ilu_fill_factor": 40, "maxiter": 4000}

    # Still works (backward compatible) but warns, for both solve and solve_stats.
    with pytest.warns(DeprecationWarning, match="ichol"):
        x = sparse.solve(a, b, **kwargs)
    assert np.linalg.norm(a @ x - b) / np.linalg.norm(b) < 1e-7
    with pytest.warns(DeprecationWarning, match="ichol"):
        sparse.solve_stats(a, b, **kwargs)


@pytest.mark.sparse
def test_sparse_solve_cg_ilu_failure_points_to_ichol():
    a = _laplacian_2d(30)
    b = np.ones(a.shape[0])
    with pytest.warns(DeprecationWarning):
        with pytest.raises(RuntimeError, match="ichol"):
            sparse.solve(a, b, method="cg", preconditioner="ilu")


@pytest.mark.sparse
def test_sparse_solve_bicgstab_respects_tol_maxiter():
    rng = np.random.default_rng(16)
    a = sp.random(35, 35, density=0.1, format="csc", random_state=16) + 8.0 * sp.eye(35, format="csc")
    b = rng.standard_normal((35, 2))

    x = sparse.solve(a, b, method="bicgstab", tol=1e-10, maxiter=4000)
    resid = np.linalg.norm(a @ x - b) / np.linalg.norm(b)
    assert resid < 1e-7


def _upwind_convection_diffusion(n=2000, eps=1e-2):
    h = 1.0 / (n + 1)
    diffusion = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n)) * (eps / h**2)
    convection = sp.diags([-1.0, 1.0], [-1, 0], shape=(n, n)) / h
    return (diffusion + convection).tocsc(), np.ones(n)


@pytest.mark.sparse
def test_sparse_solve_bicgstab_checks_true_residual():
    # Regression: BiCGSTAB's recursively updated residual drifts from the true one on this
    # nonnormal problem; it used to return with ||Ax - b|| / ||b|| ~ 1.8e-3 despite tol=1e-10.
    a, b = _upwind_convection_diffusion()
    x = sparse.solve(a, b, method="bicgstab", tol=1e-10, maxiter=20000)
    assert np.linalg.norm(a @ x - b) / np.linalg.norm(b) < 1e-9


@pytest.mark.sparse
def test_sparse_solve_bicgstab_unreachable_tol_raises():
    # The true residual of this problem bottoms out around 1e-11 in double precision, so BiCGSTAB's
    # residual estimate can reach tol=1e-14 but the answer can't: that must raise, not return.
    a, b = _upwind_convection_diffusion()
    with pytest.raises(RuntimeError, match="true relative residual"):
        sparse.solve(a, b, method="bicgstab", tol=1e-14, maxiter=20000)


@pytest.mark.sparse
def test_sparse_solve_iterative_zero_rhs():
    a = _laplacian_2d(10)
    b = np.zeros(100)
    npt.assert_array_equal(sparse.solve(a, b, method="bicgstab"), b)
    npt.assert_array_equal(sparse.solve(a, b, method="cg", preconditioner="ichol"), b)
    assert sparse.solve_stats(a, b, method="cg")["residual"] == 0.0


@pytest.mark.sparse
def test_sparse_factorized_solve():
    rng = np.random.default_rng(14)
    a = sp.random(25, 25, density=0.1, format="csc", random_state=14)
    a = a + 9.0 * sp.eye(25, format="csc")
    b = rng.standard_normal((25, 1))

    fac = sparse.factorize(a)
    x = fac.solve(b)
    resid = np.linalg.norm(a @ x - b) / np.linalg.norm(b)
    assert resid < 1e-9
