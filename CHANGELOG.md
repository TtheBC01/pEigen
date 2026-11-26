# Changelog

All notable changes to pEigen are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `SparseCholesky.L()`, `D()` and `P()`: the factors as SciPy CSC matrices and the fill-reducing
  permutation as an index array, with `(A + shift*I)[p][:, p] == L @ D @ L.T` (scikit-sparse's
  convention) for every mode. `L` stores its diagonal (unit except for `llt`); `D` is diagonal, has
  2x2 blocks for `multifrontal`, and is the identity for `llt`. This enables sampling from a
  GMRF / Gaussian process given its precision matrix, and triangular solves with `L`.
- `sparse.solve(..., method="cg", preconditioner="ichol")`: zero-fill incomplete Cholesky (Eigen
  `IncompleteCholesky`, in the matrix's own ordering), a symmetric preconditioner suited to CG. On
  2D 5-point Laplacians it roughly halves CG's iteration count (100x100: 186 → 77).
- `sparse.solve_stats` also returns `"residual"`, the true relative residual
  `‖b − Ax‖ / ‖b‖` of the solution (`"error"` remains Eigen's recursive estimate).

### Fixed

- BiCGSTAB (and CG) could report success while the true residual was far above `tol`: Eigen
  stops on a recursively updated residual that can drift from `b − Ax`. For example, BiCGSTAB on
  a 1D upwind convection–diffusion problem with `tol=1e-10` returned a solution with
  `‖Ax − b‖ / ‖b‖ ≈ 1.8e-3`. `sparse.solve` and `solve_stats` now recompute the true residual
  after every iterative solve; if it is more than 10x above `tol` the solver restarts from the
  current iterate (at most 5 times, within the same `maxiter` budget), and raises `RuntimeError`
  if the residual still misses. A `tol` that double precision can't reach for the problem now
  raises instead of returning a less accurate answer.

### Deprecated

- `preconditioner="ilu"` with `method="cg"` emits a `DeprecationWarning` and will be removed in
  3.0. `IncompleteLUT` is not symmetric, so CG can fail with it: at the default
  `ilu_fill_factor=10` it did not converge on 2D Laplacians of 30², 60² or 100² unknowns, where
  plain CG takes 54–186 iterations. It still works as before (a large fill factor makes it
  converge); its non-convergence error now points to `"ichol"`.

## [2.1.0] - 2026-11-25

### Added

- `sparse.cholesky(..., mode="multifrontal")` (experimental): Eigen `MultifrontalLDLT`, a pivoted
  `L D L^T` built from dense frontal matrices. Faster on high-fill problems (3D meshes) and stable
  on symmetric indefinite matrices (1x1 / 2x2 pivots). Supports `shift`, `refactorize`, `slogdet`.
  Solves with several right-hand sides use a supernodal blocked kernel (dense TRSM/GEMM on large
  fronts, row-contiguous updates on small ones), built from `L` on the first such solve after each
  factorization: 32-RHS solves are 2-4x faster than Eigen's own multifrontal solve.
- `SparseCholesky.inertia`: `(positive, negative, zero)` eigenvalue counts from the pivots of `D`,
  for every mode.

### Fixed

- `decomp.QR(a, mode="complete").solve(b)` raised `LinAlgError` for tall `a`; it now returns the
  least-squares solution, like `mode="reduced"`. `QR.solve` also checks the shape of `b`, and wide
  `a` raises a `ValueError` that points to `SVD.solve`.

## [2.0.0] - 2026-11-24

A ground-up rewrite. pEigen is now a set of functions that take and return NumPy arrays and SciPy
sparse matrices, built with pybind11 and linked against BLAS/LAPACK, instead of the Boost.Python
matrix classes of 1.0. **No 1.0 code will run unchanged**; see *Migrating from 1.0* below.

This is the first release on PyPI. The distribution is named **`peigentux`** (`pip install peigentux`);
the import name is still `peigen`. The unrelated `peigen` project on PyPI is not this package.

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
  `to_dense`, `from_coo`. SciPy is an optional dependency (`pip install "peigentux[sparse]"`).
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

### Migrating from 1.0

| 1.0 | 2.0 |
|-------|-----|
| `import libpeigen as peigen` | `import peigen` (and `from peigen import linalg, sparse, decomp`) |
| `peigen.dense_matrix(data, rows, cols)` | a NumPy array, e.g. `np.array(data).reshape(rows, cols)` |
| `peigen.sparse_matrix(...)` | a SciPy sparse matrix, e.g. `scipy.sparse.csc_matrix(...)` |
| matrix methods (`.transpose()`, `.norm()`, element access, operators) | NumPy/SciPy operations, or `peigen.linalg.norm` |
| decomposition methods on matrix objects | `linalg.svd`, `linalg.qr`, `linalg.eigh`, or the `decomp` classes |
| Boost.Python build | pybind11 + scikit-build-core; `pip install peigentux` |

## [1.0.0] and earlier

Boost.Python bindings exposing `dense_matrix` / `sparse_matrix` objects and decomposition methods
(module `libpeigen`), built from source only and never published to PyPI. Not maintained.

[Unreleased]: https://github.com/TtheBC01/pEigen/compare/v2.1.0...HEAD
[2.1.0]: https://github.com/TtheBC01/pEigen/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/TtheBC01/pEigen/releases/tag/v2.0.0
