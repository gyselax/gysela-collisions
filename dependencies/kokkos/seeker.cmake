message(STATUS "[koliop] Seeking Kokkos implementation.")

option(koliop_USE_SYSTEM_PROVIDED_KOKKOS "Download Kokkos, compiles it, use it. Does not rely on a system installation." OFF)

# NOTE: It is a *bit* unfortunate that Kokkos relies on CMAKE_CXX_STANDARD
# instead of a less pervasive set_target_properties(... CXX_STANDARD ...).
# NOTE: Kokkos option provided in the toolchains.

if(NOT koliop_USE_SYSTEM_PROVIDED_KOKKOS)
    include(FetchContent)
    # cmake_policy(SET CMP0135 NEW)
    FetchContent_Declare(Kokkos
                         SYSTEM                # Requires CMake v3.25
                         OVERRIDE_FIND_PACKAGE # Requires CMake v3.24
                         # URL https://github.com/kokkos/kokkos/archive/refs/tags/3.7.01.tar.gz
                         # URL_HASH "SHA512=97cb7e05aec145ba9adae30b20ae3138a532db3a56cdd5e092d8f9c1e1788efe7a6f721282e2a5c5bf9c0da4c53d28f30b27fcb546b68aa8eaa589a362eb40c1"
                         URL https://github.com/kokkos/kokkos/archive/refs/tags/4.0.00.tar.gz
                         URL_HASH "SHA512=c205d81a0556d39cd91a38254818f9c698340c313d4e33fea8e18ad9cdc0b5465585996f3224563efa2e40e9b62e676e2baf67b221eb8621d2dfa4da5282428d")
    FetchContent_MakeAvailable(Kokkos)
else()
    # NOTE: With CMake 3.24, we can jut ask FetchContent_Declare with
    # OVERRIDE_FIND_PACKAGE and put the find_package in the main control flow
    # (not in the else).
    # https://cmake.org/cmake/help/latest/module/FetchContent.html#integrating-with-find-package
    find_package(Kokkos 4.0.0 REQUIRED)
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
