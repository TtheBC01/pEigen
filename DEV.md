# Developing pEigen

## Prerequisites

- Python 3.10+
- CMake 3.18+
- A C++17 compiler
- BLAS/LAPACK (optional but recommended): Accelerate is used automatically on macOS. On Linux,
  install OpenBLAS (e.g. `sudo apt-get install libopenblas-dev`) and build with
  `CMAKE_ARGS="-DBLA_VENDOR=OpenBLAS"`; without it the build falls back to Eigen-only kernels.
  Set `PEIGEN_REQUIRE_LAPACK=1` when running tests to fail if LAPACK was not linked.
- Git submodules initialized (Eigen headers)

## Setup

```bash
git clone https://github.com/<you>/pEigen.git
cd pEigen
git submodule update --init --recursive
python -m pip install --upgrade pip
python -m pip install -e .[test]
```

## Running tests

```bash
pytest --cov=peigen --cov-report=term-missing -m "not benchmark"
```

Run full suite including slow tests:

```bash
pytest
```

## Benchmarks (optional)

Install optional dependencies:

```bash
python -m pip install -e .[benchmark,sparse]
```

Run benchmark scripts:

```bash
python benchmarks/bench_dense.py
python benchmarks/bench_sparse.py
python benchmarks/bench_cholesky.py          # add --quick for a short run
# or
scripts/bench_report.sh
```

`bench_cholesky.py` also compares against CHOLMOD when scikit-sparse is installed. It needs SuiteSparse and builds from source on most platforms:

```bash
brew install suite-sparse          # or your platform's SuiteSparse package
SUITESPARSE_INCLUDE_DIR=/opt/homebrew/include/suitesparse \
SUITESPARSE_LIBRARY_DIR=/opt/homebrew/lib \
python -m pip install scikit-sparse
```

CHOLMOD uses OpenMP and a multithreaded BLAS; for a single-core comparison, run with `OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.

## Build wheel locally

```bash
python -m pip install build
python -m build
```

## CI/CD

- PR and `main` branch testing is in `.github/workflows/ci.yml`.
- Tagged releases (`v*`) trigger `.github/workflows/release.yml`:
  - builds wheels (CPython 3.10-3.15; Linux x86_64/aarch64, macOS arm64/x86_64)
  - builds sdist
  - publishes to PyPI as `peigentux` (Trusted Publishing; see `RELEASING.md`)
  - creates GitHub release

## OpenMP

OpenMP is best-effort and optional. Release wheels default to `PEIGEN_USE_OPENMP=OFF` for portability.

For local experiments:

```bash
CMAKE_ARGS="-DPEIGEN_USE_OPENMP=ON" python -m pip install -e .
```

## Website

`website/` is the static site for [peigentux.com](https://peigentux.com): plain HTML, CSS and
JavaScript with no build step. Edit the files directly and preview them with any static server:

```bash
python -m http.server 8765 --directory website
```

- Each page repeats the shared header and footer, so a nav change touches every `*.html` file.
- Python code blocks marked `data-example` are executed by `tests/test_website_examples.py`, so
  keep them self-contained and runnable. The "Output" shown under each example is static text.
- Benchmark figures come from the scripts in `benchmarks/`. The sparse Cholesky chart reads
  `website/assets/bench-cholesky-data.js`; the dense and sparse tables are static HTML. Refresh
  them (and `lastmod` in `sitemap.xml`) after re-running the benchmarks.
- `.github/workflows/pages.yml` deploys `website/` to GitHub Pages on every push to `main` that
  touches it. One-time setup: Settings → Pages → Source "GitHub Actions", custom domain
  `peigentux.com`, plus DNS records at the domain registrar.
