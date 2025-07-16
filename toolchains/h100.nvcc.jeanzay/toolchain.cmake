set(CMAKE_CXX_COMPILER "${CMAKE_CURRENT_LIST_DIR}/nvcc_wrapper")
# set(CMAKE_Fortran_COMPILER <none>)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# Kokkos options:

set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Defined if the CUDA execution space is enabled.")
set(Kokkos_ENABLE_CUDA_CONSTEXPR ON CACHE BOOL "")
set(Kokkos_ARCH_HOPPER90 ON CACHE BOOL "Enable support for Nvidia H100.")

set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Defined if the OpenMP execution space is enabled.")
# set(Kokkos_ARCH_AVX2 ON CACHE BOOL "Optimize for AVX2 architecture (HOST).")

# Kokkos kernels options:

set(KokkosKernels_ENABLE_ALL_COMPONENTS OFF CACHE BOOL "")
set(KokkosKernels_ENABLE_COMPONENT_BATCHED ON CACHE BOOL "")
set(KokkosKernels_ENABLE_COMPONENT_BLAS ON CACHE BOOL "")
set(KokkosKernels_ADD_DEFAULT_ETI OFF CACHE BOOL "")

# Koliop options:

set(koliop_TOOLCHAIN_NAME "h100.nvcc.jeanzay")

set(koliop_ASSUME_INPUT_BUFFERS_ARE_DEVICE_COMPATIBLE OFF CACHE BOOL "")

# The rest is optional:

# Show spill, S/VGPR usage, scratch size and occupancy. -Rpass-analysis=kernel-resource-usage

set(koliop_FLAGS_DEBUG          -g -Og -ggdb -fno-omit-frame-pointer -fno-optimize-sibling-calls)
set(koliop_FLAGS_MINSIZEREL     -Os)
set(koliop_FLAGS_RELEASE        -g -O3)
set(koliop_FLAGS_RELWITHDEBINFO -g -O3 -fno-omit-frame-pointer -fno-optimize-sibling-calls)

