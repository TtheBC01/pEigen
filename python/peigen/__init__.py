"""pEigen: Eigen-backed routines for NumPy and SciPy."""

from __future__ import annotations

from . import _core, decomp, linalg, sparse


def build_config() -> dict:
    """Return compile-time build configuration for this extension."""
    return dict(_core.build_config())


def show_build_config() -> None:
    """Print a human-readable runtime build configuration report."""
    cfg = build_config()
    print("pEigen build configuration")
    print("--------------------------")
    print(f"version:               {cfg['version']}")
    print(f"build_type:            {cfg['build_type']}")
    print(f"blas_enabled:          {cfg['blas_enabled']}")
    print(f"blas_backend:          {cfg['blas_backend']}")
    print(f"lapack_enabled:        {cfg['lapack_enabled']}")
    print(f"lapack_backend:        {cfg['lapack_backend']}")
    print(f"openmp_enabled:        {cfg['openmp_enabled']}")
    print(f"vectorization_enabled: {cfg['vectorization_enabled']}")
    print(f"eigen_mpl2_only:       {cfg['eigen_mpl2_only']}")


__all__ = ["linalg", "sparse", "decomp", "build_config", "show_build_config"]
