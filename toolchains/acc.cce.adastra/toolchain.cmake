set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/installation")

set(CMAKE_CXX_COMPILER CC)
set(CMAKE_Fortran_COMPILER ftn)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release) # Debug, Release, RelWithDebInfo and MinSizeRel

# Kokkos options:

set(Kokkos_ENABLE_HIP ON CACHE BOOL "Defined if the HIP execution space is enabled.")
set(Kokkos_ENABLE_HIP_MULTIPLE_KERNEL_INSTANTIATIONS ON CACHE BOOL "If defined, multiple kernel versions are instantiated potentially improving run time.")
set(Kokkos_ARCH_VEGA90A ON CACHE BOOL "Enable support for AMD GPU MI200 series (GFX90A).")

set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Defined if the OpenMP execution space is enabled.")
set(Kokkos_ARCH_ZEN3 ON CACHE BOOL "Optimize for AMD Zen3 architecture (HOST).")

# Koliop options:

set(koliop_TOOLCHAIN_NAME "acc.cce.adastra")
set(koliop_BUILD_FORTRAN_INTERFACE OFF CACHE BOOL "")

# The rest is optional:

set(koliop_FLAGS_BASE   -Wall -Wextra -Wpedantic -Wcast-align -Wctor-dtor-privacy -Wdouble-promotion -Wformat=2 -Winit-self -Wmissing-include-dirs -Woverloaded-virtual -Wredundant-decls -Wsign-promo -Wstrict-aliasing -Wdisabled-optimization -Wsuggest-override -Wswitch-default -Wtautological-compare -Wpacked -Wunreachable-code -Wcast-qual -Wold-style-cast -Wswitch-enum -Wundef) # -Wconversion

set(koliop_FLAGS_DEBUG           -Og -fno-omit-frame-pointer -fno-optimize-sibling-calls)
set(koliop_FLAGS_MINSIZEREL      -Os)
set(koliop_FLAGS_RELEASE         -O3)
set(koliop_FLAGS_RELWITHDEBINFO  -O3 -fno-omit-frame-pointer -fno-optimize-sibling-calls)

# TODO(Etienne M): The following flags may have a positive performance impact:
# -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false
# See: https://github.com/ROCm-Developer-Tools/HIP/blob/develop/hip-lang-config.cmake.in

set(koliop_LINKER_FLAGS_DEBUG          ${koliop_FLAGS_DEBUG})
set(koliop_LINKER_FLAGS_MINSIZEREL     ${koliop_FLAGS_MINSIZEREL})
set(koliop_LINKER_FLAGS_RELEASE        ${koliop_FLAGS_RELEASE})
set(koliop_LINKER_FLAGS_RELWITHDEBINFO ${koliop_FLAGS_RELWITHDEBINFO})
