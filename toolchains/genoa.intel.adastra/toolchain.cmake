set(CMAKE_CXX_COMPILER CC)
set(CMAKE_Fortran_COMPILER ftn)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# Kokkos options:

set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Defined if the OpenMP execution space is enabled.")
set(Kokkos_ARCH_ZEN4 ON CACHE BOOL "Optimize for AMD Zen4 architecture (HOST).")

# Kokkos kernels options:

set(KokkosKernels_ENABLE_ALL_COMPONENTS OFF CACHE BOOL "")
set(KokkosKernels_ENABLE_COMPONENT_BATCHED ON CACHE BOOL "")
set(KokkosKernels_ENABLE_COMPONENT_BLAS ON CACHE BOOL "")
set(KokkosKernels_ADD_DEFAULT_ETI OFF CACHE BOOL "")

# Koliop options:

set(koliop_TOOLCHAIN_NAME "genoa.intel.adastra")

# The rest is optional:

set(koliop_FLAGS_BASE -Wall -Wextra)

set(koliop_FLAGS_DEBUG          -g -O1)
set(koliop_FLAGS_MINSIZEREL     -Os)
set(koliop_FLAGS_RELEASE        -g -O3 -xHOST -inline-factor=1000)
set(koliop_FLAGS_RELWITHDEBINFO -g -O3)

# set(koliop_LINKER_FLAGS_DEBUG          )
# set(koliop_LINKER_FLAGS_MINSIZEREL     )
# set(koliop_LINKER_FLAGS_RELEASE        )
# set(koliop_LINKER_FLAGS_RELWITHDEBINFO )
