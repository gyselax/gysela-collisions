module purge

module load cmake

module load cpe/23.05
module load craype-x86-genoa
# module load PrgEnv-cray
# We use the ROCm compiler for a CPU build. We basically, just do not use the
# HIP compiling capabilities of clang, which is perfectly fine.
module load PrgEnv-amd
module load amd/5.5.1

module list
