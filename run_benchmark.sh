#!/usr/bin/env bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mpirun --bind-to core -n "$1" python3 benchmark.py
