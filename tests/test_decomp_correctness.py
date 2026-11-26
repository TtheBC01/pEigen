import numpy as np
import numpy.testing as npt
import pytest

from peigen import decomp

sp = pytest.importorskip("scipy.sparse")


def test_svd_object_solves_least_squares():
    rng = np.random.default_rng(20)
    a = rng.standard_normal((30, 12))
    b = rng.standard_normal((30,))

    svd = decomp.SVD(a)
    x = svd.solve(b)
    x_ref, *_ = np.linalg.lstsq(a, b, rcond=None)

    npt.assert_allclose(x, x_ref, rtol=1e-9, atol=1e-9)


def test_qr_object_solves_square_system():
    rng = np.random.default_rng(21)
    a = rng.standard_normal((18, 18)) + 5 * np.eye(18)
    b = rng.standard_normal((18,))

    qr = decomp.QR(a)
    x = qr.solve(b)
    npt.assert_allclose(a @ x, b, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("mode", ["reduced", "complete"])
@pytest.mark.parametrize("shape", [(18, 18), (30, 12)], ids=["square", "tall"])
@pytest.mark.parametrize("nrhs", [None, 3])
def test_qr_object_solve_modes(mode, shape, nrhs):
    # Regression: mode="complete" on a tall matrix used to pass the non-square R to
    # numpy.linalg.solve and raise LinAlgError.
    rng = np.random.default_rng(23)
    a = rng.standard_normal(shape)
    b = rng.standard_normal(shape[0] if nrhs is None else (shape[0], nrhs))
    x = decomp.QR(a, mode=mode).solve(b)
    x_ref, *_ = np.linalg.lstsq(a, b, rcond=None)
    assert x.shape == x_ref.shape
    npt.assert_allclose(x, x_ref, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("mode", ["reduced", "complete"])
def test_qr_object_solve_rejects_wide_and_bad_rhs(mode):
    rng = np.random.default_rng(24)
    with pytest.raises(ValueError, match="SVD.solve"):
        decomp.QR(rng.standard_normal((5, 9)), mode=mode).solve(np.ones(5))
    qr = decomp.QR(rng.standard_normal((9, 5)), mode=mode)
    with pytest.raises(ValueError, match=r"shape \(9,\)"):
        qr.solve(np.ones(5))


def test_sparse_factorized_object():
    rng = np.random.default_rng(22)
    a = sp.random(18, 18, density=0.15, format="csc", random_state=22)
    a = a + 7 * sp.eye(18, format="csc")
    b = rng.standard_normal((18, 2))

    fac = decomp.SparseFactorized(a)
    x = fac.solve(b)
    residual = np.linalg.norm(a @ x - b) / np.linalg.norm(b)
    assert residual < 1e-9
