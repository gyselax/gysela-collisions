set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/installation")

set(CMAKE_CXX_COMPILER "${CMAKE_CURRENT_LIST_DIR}/nvcc_wrapper")
set(CMAKE_Fortran_COMPILER gfortran)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release) # Debug, Release, RelWithDebInfo and MinSizeRel

# Kokkos options:

set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Defined if the CUDA execution space is enabled.")
set(Kokkos_ENABLE_CUDA_CONSTEXPR ON CACHE BOOL "")
set(Kokkos_ARCH_VOLTA70 ON CACHE BOOL "Enable support for Nvidia V100.")

set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Defined if the OpenMP execution space is enabled.")
set(Kokkos_ARCH_AVX2 ON CACHE BOOL "Optimize for AVX2 architecture (HOST).")

# Koliop options:

set(koliop_TOOLCHAIN_NAME "v100.nvcc.persee")

set(koliop_FENCE_ON_OPERATOR_EXIT OFF)
set(koliop_ASSUME_INPUT_BUFFERS_ARE_DEVICE_COMPATIBLE OFF)

# The rest is optional:

set(koliop_FLAGS_BASE -Wall -Wextra -Wpedantic -Wcast-align -Wctor-dtor-privacy -Wdouble-promotion -Wformat=2 -Winit-self -Woverloaded-virtual -Wredundant-decls -Wsign-promo -Wstrict-aliasing -Wdisabled-optimization -Wsuggest-override -Wswitch-default -Wtautological-compare -Wpacked -Wunreachable-code -Wcast-qual -Wold-style-cast -Wswitch-enum -Wundef) # -Wconversion -Wmissing-include-dirs

set(koliop_FLAGS_FP -fno-math-errno -fno-trapping-math -ffp-contract=fast)

# Show spill, S/VGPR usage, scratch size and occupancy. -Rpass-analysis=kernel-resource-usage

set(koliop_FLAGS_DEBUG          -g -Og -ggdb -fno-omit-frame-pointer -fno-optimize-sibling-calls)
set(koliop_FLAGS_MINSIZEREL     -Os)
set(koliop_FLAGS_RELEASE        -g -O3)
set(koliop_FLAGS_RELWITHDEBINFO -g -O3 -fno-omit-frame-pointer -fno-optimize-sibling-calls)

# set(koliop_LINKER_FLAGS_DEBUG          ${koliop_FLAGS_DEBUG})
# set(koliop_LINKER_FLAGS_MINSIZEREL     ${koliop_FLAGS_MINSIZEREL})
# set(koliop_LINKER_FLAGS_RELEASE        ${koliop_FLAGS_RELEASE})
# set(koliop_LINKER_FLAGS_RELWITHDEBINFO ${koliop_FLAGS_RELWITHDEBINFO})
