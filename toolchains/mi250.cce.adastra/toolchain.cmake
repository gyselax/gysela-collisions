set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/installation")

set(CMAKE_CXX_COMPILER CC)
set(CMAKE_Fortran_COMPILER ftn)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# Kokkos options:

set(Kokkos_ENABLE_HIP ON CACHE BOOL "Defined if the HIP execution space is enabled.")
set(Kokkos_ENABLE_HIP_MULTIPLE_KERNEL_INSTANTIATIONS ON CACHE BOOL "If defined, multiple kernel versions are instantiated potentially improving run time.")
set(Kokkos_ARCH_VEGA90A ON CACHE BOOL "Enable support for AMD GPU MI200 series (GFX90A).")

set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Defined if the OpenMP execution space is enabled.")
set(Kokkos_ARCH_ZEN3 ON CACHE BOOL "Optimize for AMD Zen3 architecture (HOST).")

# Kokkos kernels options:

set(KokkosKernels_ENABLE_ALL_COMPONENTS OFF CACHE BOOL "")
set(KokkosKernels_ENABLE_COMPONENT_BATCHED ON CACHE BOOL "")
set(KokkosKernels_ENABLE_COMPONENT_BLAS ON CACHE BOOL "")

# Koliop options:

set(koliop_TOOLCHAIN_NAME "mi250.cce.adastra")

set(koliop_ENABLE_LTO OFF CACHE BOOL "")
set(koliop_ASSUME_INPUT_BUFFERS_ARE_DEVICE_COMPATIBLE OFF CACHE BOOL "")

# The rest is optional:

set(koliop_FLAGS_BASE -Wall -Wextra -Wpedantic -Wcast-align -Wctor-dtor-privacy -Wdouble-promotion -Wformat=2 -Winit-self -Woverloaded-virtual -Wredundant-decls -Wsign-promo -Wstrict-aliasing -Wdisabled-optimization -Wsuggest-override -Wswitch-default -Wtautological-compare -Wpacked -Wunreachable-code -Wcast-qual -Wold-style-cast -Wswitch-enum -Wundef) # -Wconversion -Wmissing-include-dirs

set(koliop_FLAGS_FP -fno-math-errno -fno-trapping-math -ffp-contract=fast)

# Show spill, S/VGPR usage, scratch size and occupancy. -Rpass-analysis=kernel-resource-usage

set(koliop_FLAGS_DEBUG          -g -Og -ggdb -fno-omit-frame-pointer -fno-optimize-sibling-calls)
set(koliop_FLAGS_MINSIZEREL     -Os)
set(koliop_FLAGS_RELEASE        -g -O3 "SHELL:-mllvm -amdgpu-early-inline-all=true" "SHELL:-mllvm -amdgpu-function-calls=false")
set(koliop_FLAGS_RELWITHDEBINFO -g -O3 "SHELL:-mllvm -amdgpu-early-inline-all=true" "SHELL:-mllvm -amdgpu-function-calls=false" -fno-omit-frame-pointer -fno-optimize-sibling-calls)

# FIXME: https://gitlab.kitware.com/cmake/cmake/-/issues/24402 https://gitlab.kitware.com/cmake/cmake/-/issues/25100
set(koliop_LINKER_FLAGS_DEBUG          -fopenmp)
set(koliop_LINKER_FLAGS_MINSIZEREL     -fopenmp)
set(koliop_LINKER_FLAGS_RELEASE        -fopenmp)
set(koliop_LINKER_FLAGS_RELWITHDEBINFO -fopenmp)
