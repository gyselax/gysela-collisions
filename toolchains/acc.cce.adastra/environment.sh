#!/bin/bash

module purge

module load craype-accel-amd-gfx90a craype-x86-trento
module load PrgEnv-cray/8.3.3
module load amd-mixed/5.2.3

# module load perftools-base/22.09.0
# module load perftools-lite-gpu

module list
