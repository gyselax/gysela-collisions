message(STATUS "[koliop] Seeking Kokkos implementation.")

option(koliop_USE_SYSTEM_PROVIDED_KOKKOS        "Look for a system provided installation instead of downloading Kokkos and compiling it." OFF)
option(koliop_USE_SYSTEM_PROVIDED_KOKKOS_KERNEL "Look for a system provided installation instead of downloading Kokkos Kernel and compiling it." OFF)

# NOTE: It is a *bit* unfortunate that Kokkos relies on CMAKE_CXX_STANDARD
# instead of a less pervasive set_target_properties(... CXX_STANDARD ...).
# NOTE: Kokkos options provided in the toolchains.

if(NOT koliop_USE_SYSTEM_PROVIDED_KOKKOS)
    # cmake_policy(SET CMP0135 NEW)
    FetchContent_Declare(Kokkos
                         SYSTEM                # For CMake v3.25 and up.
                         OVERRIDE_FIND_PACKAGE # For CMake v3.24 and up.
                         # URL https://github.com/kokkos/kokkos/archive/refs/tags/3.7.01.tar.gz
                         # URL_HASH "SHA512=97cb7e05aec145ba9adae30b20ae3138a532db3a56cdd5e092d8f9c1e1788efe7a6f721282e2a5c5bf9c0da4c53d28f30b27fcb546b68aa8eaa589a362eb40c1"
                         # URL https://github.com/kokkos/kokkos/archive/refs/tags/4.0.00.tar.gz
                         # URL_HASH "SHA512=c205d81a0556d39cd91a38254818f9c698340c313d4e33fea8e18ad9cdc0b5465585996f3224563efa2e40e9b62e676e2baf67b221eb8621d2dfa4da5282428d"
                         URL https://github.com/kokkos/kokkos/archive/refs/tags/4.0.01.tar.gz
                         URL_HASH "SHA512=cf6742f4be777b52673858b49cf68fdf335e760509688483e62ef41af5026070f69dcf39d010134c85dc6c4868f82a38900872be11ed43190432c8530b90deee")
    FetchContent_MakeAvailable(Kokkos)
else()
    # NOTE: With CMake 3.24, we can jut ask FetchContent_Declare with
    # OVERRIDE_FIND_PACKAGE and put the find_package in the main control flow
    # (not in the else).
    # https://cmake.org/cmake/help/latest/module/FetchContent.html#integrating-with-find-package
    find_package(Kokkos 4.0.1 REQUIRED)
endif()

if(NOT koliop_USE_SYSTEM_PROVIDED_KOKKOS_KERNEL)
    set(KokkosKernels_ENABLE_ALL_COMPONENTS OFF CACHE BOOL "")
    set(KokkosKernels_ENABLE_COMPONENT_BATCHED ON CACHE BOOL "")
    set(KokkosKernels_ENABLE_COMPONENT_BLAS ON CACHE BOOL "")

    # cmake_policy(SET CMP0135 NEW)
    FetchContent_Declare(KokkosKernels
                         SYSTEM                # For CMake v3.25 and up.
                         OVERRIDE_FIND_PACKAGE # For CMake v3.24 and up.
                         # URL https://github.com/kokkos/kokkos-kernels/archive/refs/tags/4.0.00.tar.gz
                         # URL_HASH "SHA512=97a2c24150eb2fd64599337248f77b3e4350164fee01656cf3d71c55dfd5733657e53c70b31dc2872c1a6d63ce679e1c61ce45a236f8ac5927ae6eac85f9cdfe"
                         URL https://github.com/kokkos/kokkos-kernels/archive/refs/tags/4.0.01.tar.gz
                         URL_HASH "SHA512=41f93513bde4a71baae5572830d0cebf1d5caa3a062ac1d2c159c0853791a16f587c27b1c190a2a8e582513b009023528b16d16c73de215ca15675abb9444ad7")
    FetchContent_MakeAvailable(KokkosKernels)
else()
    # NOTE: Same remark as the previous find_package.
    find_package(KokkosKernels 4.0.1 REQUIRED)
endif()

function(koliop_target_add_kokkos target_name)
    target_link_libraries("${target_name}"
                          PRIVATE Kokkos::kokkos)

    if(koliop_TOOLCHAIN_NAME MATCHES "acc.cce.adastra")
        # WORKAROUND:
        # https://gitlab.kitware.com/cmake/cmake/-/issues/24402
        target_link_options("${target_name}"
                            PUBLIC -fopenmp)
    endif()
endfunction()

function(koliop_target_add_kokkos_kernels target_name)
    target_link_libraries("${target_name}"
                          PRIVATE Kokkos::kokkoskernels)
endfunction()
