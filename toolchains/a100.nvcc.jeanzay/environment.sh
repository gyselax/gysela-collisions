#!/bin/bash

module purge

module load cmake/3.25.2 ninja/1.10.0
# module load nvidia-compilers/25.1
module load cuda/12.8.0

# module load openmpi/4.1.1-cuda
# module load hdf5/1.12.0-mpi-cuda

module list
