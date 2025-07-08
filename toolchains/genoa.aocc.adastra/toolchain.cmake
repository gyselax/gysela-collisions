set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_Fortran_COMPILER flang)
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

set(koliop_TOOLCHAIN_NAME "genoa.aocc.adastra")

# The rest is optional:

set(koliop_FLAGS_BASE -Wall -Wextra -Wpedantic -Wcast-align -Wctor-dtor-privacy -Wdouble-promotion -Wformat=2 -Winit-self -Woverloaded-virtual -Wredundant-decls -Wsign-promo -Wstrict-aliasing -Wdisabled-optimization -Wsuggest-override -Wswitch-default -Wtautological-compare -Wpacked -Wunreachable-code -Wcast-qual -Wold-style-cast -Wswitch-enum -Wundef) # -Wconversion -Wmissing-include-dirs

set(koliop_FLAGS_DEBUG          -g -Og -ggdb -fno-omit-frame-pointer -fno-optimize-sibling-calls)
set(koliop_FLAGS_MINSIZEREL     -Os)
set(koliop_FLAGS_RELEASE        -g -O3 -fno-math-errno -march=znver4 -fveclib=AMDLIBM) # -Ofast -lamdlibmfast
set(koliop_FLAGS_RELWITHDEBINFO -g -O3 -fno-math-errno -march=znver4 -fveclib=AMDLIBM -fno-omit-frame-pointer -fno-optimize-sibling-calls)

set(koliop_LINKER_FLAGS_DEBUG          )
set(koliop_LINKER_FLAGS_MINSIZEREL     )
set(koliop_LINKER_FLAGS_RELEASE        -fveclib=AMDLIBM -lamdlibm) # "$ENV{AOCC_ROOT}/lib/libamdlibm.a" # NOTE: the static lib is not used.. This seems to be due to some lib ordering in CMake
set(koliop_LINKER_FLAGS_RELWITHDEBINFO -fveclib=AMDLIBM -lamdlibm) # "$ENV{AOCC_ROOT}/lib/libamdlibm.a"
