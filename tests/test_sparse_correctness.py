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


@pytest.mark.parametrize("preconditioner", ["none", "jacobi", "ilu"])
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
    if preconditioner == "ilu":
        solve_kwargs["ilu_fill_factor"] = 40
        solve_kwargs["ilu_drop_tol"] = 1e-4

    x = sparse.solve(a, b, **solve_kwargs)
    resid = np.linalg.norm(a @ x - b) / np.linalg.norm(b)
    assert resid < 1e-6

    stats = sparse.solve_stats(a, b, **solve_kwargs)
    assert stats["iterations"] >= 0
    assert stats["error"] >= 0.0


@pytest.mark.sparse
def test_sparse_solve_bicgstab_respects_tol_maxiter():
    rng = np.random.default_rng(16)
    a = sp.random(35, 35, density=0.1, format="csc", random_state=16) + 8.0 * sp.eye(35, format="csc")
    b = rng.standard_normal((35, 2))

    x = sparse.solve(a, b, method="bicgstab", tol=1e-10, maxiter=4000)
    resid = np.linalg.norm(a @ x - b) / np.linalg.norm(b)
    assert resid < 1e-7


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
