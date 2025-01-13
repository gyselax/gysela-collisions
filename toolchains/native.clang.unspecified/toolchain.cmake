set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# Kokkos options:

set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Defined if the OpenMP execution space is enabled.")
set(Kokkos_ARCH_NATIVE ON CACHE BOOL "")

# Kokkos kernels options:

set(KokkosKernels_ENABLE_ALL_COMPONENTS OFF CACHE BOOL "")
set(KokkosKernels_ENABLE_COMPONENT_BATCHED ON CACHE BOOL "")
set(KokkosKernels_ENABLE_COMPONENT_BLAS ON CACHE BOOL "")

# Koliop options:

set(koliop_TOOLCHAIN_NAME "native.clang.unspecified")

# The rest is optional:

set(koliop_FLAGS_BASE -Wall -Wextra -Wpedantic -Wcast-align -Wctor-dtor-privacy -Wdouble-promotion -Wformat=2 -Winit-self -Woverloaded-virtual -Wredundant-decls -Wsign-promo -Wstrict-aliasing -Wdisabled-optimization -Wsuggest-override -Wswitch-default -Wtautological-compare -Wpacked -Wunreachable-code -Wcast-qual -Wold-style-cast -Wswitch-enum -Wundef) # -Wconversion -Wmissing-include-dirs

set(koliop_FLAGS_DEBUG          -g -Og -ggdb -fno-omit-frame-pointer -fno-optimize-sibling-calls)
set(koliop_FLAGS_MINSIZEREL     -Os)
#SVML offers AVX512 support but needs to be linked, while libmvec is implicit.
set(koliop_FLAGS_RELEASE        -g -O3 -fno-math-errno -fno-signed-zeros -fno-trapping-math -fassociative-math -freciprocal-math -fno-rounding-math -ffp-contract=fast -fveclib=libmvec)
set(koliop_FLAGS_RELWITHDEBINFO -g -O3 -fno-omit-frame-pointer -fno-optimize-sibling-calls)
