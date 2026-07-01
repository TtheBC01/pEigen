# Developing pEigen

## Prerequisites

- Python 3.10+
- CMake 3.18+
- A C++17 compiler
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
# or
scripts/bench_report.sh
```

## Build wheel locally

```bash
python -m pip install build
python -m build
```

## CI/CD

- PR and `main` branch testing is in `.github/workflows/ci.yml`.
- Tagged releases (`v*`) trigger `.github/workflows/release.yml`:
  - builds wheels (macOS arm64/x86_64 + Linux x86_64)
  - builds sdist
  - publishes to PyPI (Trusted Publishing)
  - creates GitHub release

## OpenMP

OpenMP is best-effort and optional. Release wheels default to `PEIGEN_USE_OPENMP=OFF` for portability.

For local experiments:

```bash
CMAKE_ARGS="-DPEIGEN_USE_OPENMP=ON" python -m pip install -e .
```
