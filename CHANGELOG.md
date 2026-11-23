# Changelog

All notable changes to pEigen are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## [2.0.0] - Unreleased

A ground-up rewrite. pEigen is now a set of functions that take and return NumPy arrays and SciPy
sparse matrices, built with pybind11 and linked against BLAS/LAPACK, instead of the Boost.Python
matrix classes of 0.0.x. **No 0.0.x code will run unchanged**; see *Migrating from 0.0.x* below.

### Added

- **Sparse Cholesky**: `peigen.sparse.cholesky()` / `SparseCholesky` (Eigen `SimplicialLDLT` /
  `SimplicialLLT`, AMD or natural ordering). Supports `solve` with 1D or 2D right-hand sides,
  same-pattern `refactorize`, `slogdet` / `logdet`, a diagonal `shift`, and reading either
  triangle. Many-RHS solves use a blocked kernel. Also available as
  `sparse.solve(..., method="cholesky")` and `peigen.decomp.SparseCholesky`.
- Dense routines in `peigen.linalg`: `matmul`, `solve`, `qr`, `svd`, `eigh`, `eighvals`, `norm`,
  with `method="auto" | "lapack" | "eigen"` backend selection where applicable.
- Sparse routines in `peigen.sparse`: `spmm`, `spspmm`, `solve` (SparseLU, CG with
  none/Jacobi/ILU preconditioning, BiCGSTAB), `solve_stats`, `factorize` (reusable SparseLU),
  `to_dense`, `from_coo`. SciPy is an optional dependency (`pip install peigen[sparse]`).
- Stateful helpers in `peigen.decomp`: `SVD`, `QR`, `SparseFactorized`, `SparseCholesky`.
- `peigen.build_config()` / `show_build_config()` report the compiled-in BLAS/LAPACK backend,
  OpenMP, vectorization and Eigen version.
- Wheels for CPython 3.10-3.15 on Linux (x86_64, aarch64; OpenBLAS) and macOS (arm64, x86_64;
  Accelerate).
- Benchmarks comparing against NumPy, SciPy and (optionally) CHOLMOD in `benchmarks/`.

### Build

- Built against an Eigen master snapshot (`c7b9c4bc8`, 2026-11-22; reports `5.0.1-dev+master`),
  which includes recent SparseCore/SparseLU/SparseCholesky fixes not yet in a tagged release.
- LAPACK routines are declared by pEigen itself (LP64, gfortran hidden string lengths) instead of
  through Eigen's internal `lapack/lapack.h`.
- Package license metadata is `MIT AND MPL-2.0`: pEigen is MIT, and wheels/sdists bundle Eigen
  (MPL-2.0, built with `EIGEN_MPL2_ONLY`). Eigen's license notice ships in the wheel.

### Input handling

- Sparse inputs accept any SciPy sparse format (`*_matrix` or `*_array`), `int32` or `int64`
  indices and any real dtype; non-canonical matrices (unsorted or duplicate entries) are
  canonicalized on a copy, never in place. Matrices beyond the 32-bit index range raise
  `ValueError`.
- Complex input raises `TypeError` instead of silently discarding the imaginary part.
- `decomp.SVD.solve(b, rcond=None)`: `rcond` is relative to the largest singular value, matching
  `numpy.linalg.lstsq`.

### Migrating from 0.0.x

| 0.0.x | 2.0 |
|-------|-----|
| `import libpeigen as peigen` | `import peigen` (and `from peigen import linalg, sparse, decomp`) |
| `peigen.dense_matrix(data, rows, cols)` | a NumPy array, e.g. `np.array(data).reshape(rows, cols)` |
| `peigen.sparse_matrix(...)` | a SciPy sparse matrix, e.g. `scipy.sparse.csc_matrix(...)` |
| matrix methods (`.transpose()`, `.norm()`, element access, operators) | NumPy/SciPy operations, or `peigen.linalg.norm` |
| decomposition methods on matrix objects | `linalg.svd`, `linalg.qr`, `linalg.eigh`, or the `decomp` classes |
| Boost.Python build | pybind11 + scikit-build-core; `pip install peigen` |

## [0.0.9] and earlier

Boost.Python bindings exposing `dense_matrix` / `sparse_matrix` objects and decomposition methods
(module `libpeigen`). Not maintained.

[2.0.0]: https://github.com/TtheBC01/pEigen/releases/tag/v2.0.0
