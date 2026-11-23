"""Internal input-validation helpers."""

from __future__ import annotations

import numpy as np


def as_real_float64(a, name: str = "input") -> np.ndarray:
    """Convert ``a`` to a float64 ndarray, rejecting complex input.

    ``np.asarray(z, dtype=np.float64)`` on complex data silently drops the imaginary part (with only
    a ``ComplexWarning``), which would make every routine return a wrong answer. pEigen only supports
    real data, so fail loudly instead.
    """
    arr = np.asarray(a)
    if np.iscomplexobj(arr):
        raise TypeError(f"{name} is complex; pEigen only supports real (float64) data")
    return np.asarray(arr, dtype=np.float64)
