module purge

module load cpe/23.12
# FIXME: CPE + this module breaks CMake detection of OpenMP
# module load craype-accel-amd-gfx90a
module load craype-x86-trento
module load PrgEnv-amd
module load amd/5.7.1

module list
