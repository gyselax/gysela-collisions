set(CMAKE_CXX_COMPILER amdclang++)
set(CMAKE_Fortran_COMPILER amdflang)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# Kokkos options:

set(Kokkos_ENABLE_OPENMPTARGET ON CACHE BOOL "")
set(Kokkos_ARCH_AMD_GFX942 ON CACHE BOOL "")

set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "")
set(Kokkos_ARCH_ZEN4 ON CACHE BOOL "")

# Kokkos kernels options:

set(KokkosKernels_ENABLE_ALL_COMPONENTS OFF CACHE BOOL "")
set(KokkosKernels_ENABLE_COMPONENT_BATCHED ON CACHE BOOL "")
set(KokkosKernels_ENABLE_COMPONENT_BLAS ON CACHE BOOL "")
set(KokkosKernels_ADD_DEFAULT_ETI OFF CACHE BOOL "")

# Koliop options:

set(koliop_TOOLCHAIN_NAME "mi300a.amdomp.adastra")

set(koliop_ASSUME_INPUT_BUFFERS_ARE_DEVICE_COMPATIBLE OFF CACHE BOOL "")
set(koliop_CONSTANT_MEMORY_USAGE ON CACHE BOOL "")

# The rest is optional:

set(koliop_FLAGS_BASE -Wall -Wextra -Wpedantic -Wcast-align -Wctor-dtor-privacy -Wdouble-promotion -Wformat=2 -Winit-self -Woverloaded-virtual -Wredundant-decls -Wsign-promo -Wstrict-aliasing -Wdisabled-optimization -Wsuggest-override -Wswitch-default -Wtautological-compare -Wpacked -Wunreachable-code -Wcast-qual -Wold-style-cast -Wswitch-enum -Wundef) # -Wconversion -Wmissing-include-dirs

# Show spill, S/VGPR usage, scratch size and occupancy. -Rpass-analysis=kernel-resource-usage
# -fopenmp-target-fast is supposed to be:
#   -fopenmp-assume-no-thread-state -fopenmp-target-ignore-env-vars -fopenmp-assume-no-nested-parallelism
#    but with ROCM 6.4.0, we get better perf using -fopenmp-target-fast instead of the individual flags.

set(koliop_FLAGS_DEBUG          -g -Og -ggdb -fno-omit-frame-pointer -fno-optimize-sibling-calls)
set(koliop_FLAGS_MINSIZEREL     -Os)
set(koliop_FLAGS_RELEASE        -g -O3 -fopenmp-target-fast -fno-math-errno -fno-trapping-math -ffp-contract=fast) # -ffast-math allows the compiler to use SIMD implementation of some functions.
set(koliop_FLAGS_RELWITHDEBINFO -g -O3 -fno-omit-frame-pointer -fno-optimize-sibling-calls)
