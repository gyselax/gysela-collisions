#!/bin/bash

set -eu
set +x

make -j

export OMP_DISPLAY_AFFINITY=TRUE OMP_PROC_BIND=CLOSE

export OMP_PLACES=THREADS OMP_NUM_THREADS=16
time numactl --membind=0 --physcpubind="0-23,192-215" -- ./tests/demo_dummy_integration
export OMP_PLACES=THREADS OMP_NUM_THREADS=32
time numactl --membind=0 --physcpubind="0-23,192-215" -- ./tests/demo_dummy_integration
export OMP_PLACES=THREADS OMP_NUM_THREADS=48
time numactl --membind=0 --physcpubind="0-23,192-215" -- ./tests/demo_dummy_integration

export OMP_PLACES=CORES OMP_NUM_THREADS=8
time numactl --membind=0 --physcpubind="0-23,192-215" -- ./tests/demo_dummy_integration
export OMP_PLACES=CORES OMP_NUM_THREADS=16
time numactl --membind=0 --physcpubind="0-23,192-215" -- ./tests/demo_dummy_integration
export OMP_PLACES=CORES OMP_NUM_THREADS=24
time numactl --membind=0 --physcpubind="0-23,192-215" -- ./tests/demo_dummy_integration
