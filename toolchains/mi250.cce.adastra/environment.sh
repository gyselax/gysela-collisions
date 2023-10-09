module purge

module load cpe/23.02

# TODO(Etienne M): uncomment craype-accel-amd-gfx90a when that get fixed or we
# need to correctly link with MPI.
#
# NOTE: See this [issue](https://github.com/kokkos/kokkos/issues/6044#issuecomment-1506671062)
# if you get the following error:
# error: unsupported option '-fopenmp-targets=' for language mode 'HIP'
# TLDR: Do not load (comment) the craype-accel-amd-gfx90a so that Kokkos
# compiles. Then, load (uncomment) it.
# module load craype-accel-amd-gfx90a

module load craype-x86-trento
module load PrgEnv-cray
module load amd-mixed/5.2.0

module list
