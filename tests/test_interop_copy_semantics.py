import numpy as np

from peigen import linalg


def test_input_arrays_not_mutated():
    rng = np.random.default_rng(30)
    a = rng.standard_normal((10, 10))
    b = rng.standard_normal((10, 10))

    a_before = a.copy()
    b_before = b.copy()

    _ = linalg.matmul(a, b)

    assert np.array_equal(a, a_before)
    assert np.array_equal(b, b_before)


def test_output_is_numpy_array():
    x = linalg.matmul(np.eye(3), np.ones((3, 2)))
    assert isinstance(x, np.ndarray)
    assert x.shape == (3, 2)
