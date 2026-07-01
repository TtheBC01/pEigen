#!/usr/bin/env bash
set -euo pipefail

python benchmarks/bench_dense.py
python benchmarks/bench_sparse.py
