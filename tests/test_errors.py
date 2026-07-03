import numpy as np
import pytest

from peigen import linalg, sparse


def test_dense_matmul_dimension_error():
    with pytest.raises(ValueError):
        linalg.matmul(np.zeros((3, 2)), np.zeros((3, 4)))


def test_dense_solve_requires_square():
    with pytest.raises(ValueError):
        linalg.solve(np.zeros((2, 3)), np.zeros((2, 1)))


def test_dense_rejects_non_2d():
    with pytest.raises(ValueError):
        linalg.matmul(np.zeros((4,)), np.zeros((4, 1)))


def test_sparse_import_error_when_scipy_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError):
        sparse.spmm([[1.0]], np.array([[1.0]]))


def test_svd_invalid_method():
    a = np.eye(4)
    with pytest.raises(ValueError):
        linalg.svd(a, method="unsupported")


def test_eigh_invalid_method():
    a = np.eye(4)
    with pytest.raises(ValueError):
        linalg.eigh(a, method="unsupported")


def test_solve_invalid_method():
    with pytest.raises(ValueError):
        linalg.solve(np.eye(4), np.ones(4), method="unsupported")


def test_sparse_solve_invalid_method():
    scipy = pytest.importorskip("scipy.sparse")
    a = scipy.eye(4, format="csc")
    b = np.ones((4, 1))
    with pytest.raises(ValueError):
        sparse.solve(a, b, method="unsupported")


def test_sparse_solve_preconditioner_requires_cg():
    scipy = pytest.importorskip("scipy.sparse")
    a = scipy.eye(4, format="csc")
    b = np.ones((4, 1))
    with pytest.raises(ValueError):
        sparse.solve(a, b, method="lu", preconditioner="jacobi")
