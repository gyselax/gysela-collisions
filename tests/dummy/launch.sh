#!/bin/bash

set -eu
set +x

make -j

export OMP_DISPLAY_AFFINITY=TRUE OMP_PROC_BIND=CLOSE

export OMP_PLACES=THREADS OMP_NUM_THREADS=16
time ./tests/demo_dummy_integration
export OMP_PLACES=THREADS OMP_NUM_THREADS=32
time ./tests/demo_dummy_integration
export OMP_PLACES=THREADS OMP_NUM_THREADS=48
time ./tests/demo_dummy_integration

export OMP_PLACES=THREADS OMP_NUM_THREADS=8
time ./tests/demo_dummy_integration
export OMP_PLACES=THREADS OMP_NUM_THREADS=16
time ./tests/demo_dummy_integration
export OMP_PLACES=THREADS OMP_NUM_THREADS=24
time ./tests/demo_dummy_integration
