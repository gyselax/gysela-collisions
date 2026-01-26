# NOTES: https://gist.github.com/mbinna/c61dbb39bca0e4fb7d1f73b0d66a4fd1

cmake_minimum_required(VERSION 3.25)

################################################################################
### Default build type.
################################################################################

if(CMAKE_BUILD_TYPE OR CMAKE_CONFIGURATION_TYPES)
    message(STATUS "[libkoliop] Build type is '${CMAKE_BUILD_TYPE}', configuration types are '${CMAKE_CONFIGURATION_TYPES}'.")
else()
    message(WARNING "[libkoliop] Nor CMAKE_BUILD_TYPE or CMAKE_CONFIGURATION_TYPES are defined.")
    if (NOT (CMAKE_BUILD_TYPE OR CONFIGURATION_TYPES))
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build type." FORCE)
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
        message(STATUS "[libkoliop] No build type was defined, so we defaulted to '${CMAKE_BUILD_TYPE}'.")
    endif()
endif()

project("koliop" VERSION 0.0.22 LANGUAGES CXX)

include(CTest)

if(koliop_TOOLCHAIN_NAME)
    message(STATUS "[koliop] Using toolchain: '${koliop_TOOLCHAIN_NAME}'.")
endif()


################################################################################
### Global options. Dependency specific options are defined in the dependencies.
################################################################################

option(koliop_ASSERT_ENABLED                             "Enable assertions. This enable/disable KOLIOP_ASSERT() but not C's assert()." ON)
option(koliop_FENCE_ON_OPERATOR_EXIT                     "Fences before returning from an operator method." OFF)
option(koliop_ASSUME_INPUT_BUFFERS_ARE_DEVICE_COMPATIBLE "Should koliop assumes the input data can be read or else, intermediate copies are required." ON)
option(koliop_BUILD_FORTRAN_INTERFACE                    "Enable the koliop::fortran_interface glue code needed to interface koliop from Fortran." OFF)
option(koliop_BUILD_TESTING                              "Enable test compilation and ctest regardless of BUILD_TESTING's value." "${BUILD_TESTING}")
option(koliop_UTILITY_FLATTEN_EXPECT_POW2_EXTENTS        "Assumes that mu and vpar are going to be powers of two (speeds up indexing)." OFF)
option(koliop_CONSTANT_MEMORY_USAGE                      "Koliop will allocate scratch memory, will not release it but will reuse it for the next call." OFF)

################################################################################
### Dependencies & tooling
################################################################################

include("${CMAKE_CURRENT_LIST_DIR}/dependencies/resolver.cmake")

DependencyResolver("${CMAKE_CURRENT_SOURCE_DIR}/dependencies")


################################################################################
### Compilation, link flags.
################################################################################

if(CMAKE_BUILD_TYPE OR CMAKE_CONFIGURATION_TYPES)
    message(STATUS "[koliop] Build type is '${CMAKE_BUILD_TYPE}', configuration types are '${CMAKE_CONFIGURATION_TYPES}'.")
else()
    message(WARNING "[koliop] Nor CMAKE_BUILD_TYPE or CMAKE_CONFIGURATION_TYPES are defined.")
endif()

message(STATUS "[koliop] Assuming the compiler is '${CMAKE_CXX_COMPILER_ID}'.")

########################################
### Flags
########################################

# Specific build options for everything in this project (does not modify the
# parent configuration).
# Notably used by the tests and this project.
function(koliop_dependent_target_build_options target_name)
    # Use the flags defined in the toolchain. If the
    # koliop_[FLAGS|LINKER_FLAGS]_* are not defined, they are ignored transparently.
    set(koliop_FLAGS                            ${koliop_FLAGS_BASE}
                              $<$<CONFIG:DEBUG>:${koliop_FLAGS_DEBUG}>
                         $<$<CONFIG:MINSIZEREL>:${koliop_FLAGS_MINSIZEREL}>
                            $<$<CONFIG:RELEASE>:${koliop_FLAGS_RELEASE}>
                     $<$<CONFIG:RELWITHDEBINFO>:${koliop_FLAGS_RELWITHDEBINFO}>)

    set(koliop_LINKER_FLAGS          $<$<CONFIG:DEBUG>:${koliop_LINKER_FLAGS_DEBUG}>
                                $<$<CONFIG:MINSIZEREL>:${koliop_LINKER_FLAGS_MINSIZEREL}>
                                   $<$<CONFIG:RELEASE>:${koliop_LINKER_FLAGS_RELEASE}>
                            $<$<CONFIG:RELWITHDEBINFO>:${koliop_LINKER_FLAGS_RELWITHDEBINFO}>)

    target_compile_options("${target_name}"
                           PRIVATE ${koliop_FLAGS})

    target_link_options("${target_name}"
                        PRIVATE ${koliop_LINKER_FLAGS})

    koliop_target_enable_lto("${target_name}")
endfunction()


################################################################################
### koliop library
################################################################################

# NOTE: For now, we force the library to be static. It is often harder to
# maintain dynamic libraries. Later, we can rely on the BUILD_SHARED_LIBS
# variable and GenerateExportHeader:
# https://cmake.org/cmake/help/latest/module/GenerateExportHeader.html
add_library(koliop STATIC)
add_library("koliop::koliop" ALIAS "koliop")

set(koliop_SOURCES_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/sources")
set(koliop_INCLUDE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include")

configure_file("${koliop_INCLUDE_DIRECTORY}/KOLIOP/internal/buildspecificities.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/KOLIOP/internal/buildspecificities.h" @ONLY)

# NOTE: GLOBing file is not considered good practice. One could instead specify
# the full, verbose list of headers and sources.
file(GLOB_RECURSE koliop_SOURCES_GLOB "${koliop_SOURCES_DIRECTORY}/*.cc")

target_sources(koliop
               PRIVATE ${koliop_SOURCES_GLOB}
               PUBLIC
                 FILE_SET HEADERS
                   BASE_DIRS include ${CMAKE_CURRENT_BINARY_DIR}
                   FILES
                     ${CMAKE_CURRENT_BINARY_DIR}/KOLIOP/internal/buildspecificities.h
                     include/KOLIOP/error/status.h
                     include/KOLIOP/koliop.h
)

target_include_directories(koliop
                           PRIVATE "${koliop_SOURCES_DIRECTORY}")

# Be careful, the fortran interface will not be installed.
if(koliop_BUILD_FORTRAN_INTERFACE)
    message(STATUS "[koliop] Enabling the Fortran interface.")

    enable_language(C)
    enable_language(Fortran)

    include(FortranCInterface)

    FortranCInterface_VERIFY(CXX)

    add_library(koliop_fortran_interface STATIC)
    add_library("koliop::fortran_interface" ALIAS "koliop_fortran_interface")

    target_include_directories(koliop_fortran_interface
                               INTERFACE "${CMAKE_CURRENT_BINARY_DIR}/koliop_fortran_interface_FMOD")

    set_target_properties(koliop_fortran_interface
                          PROPERTIES Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/koliop_fortran_interface_FMOD")

    target_sources(koliop_fortran_interface
                   PRIVATE "${koliop_INCLUDE_DIRECTORY}/KOLIOP/koliop.f90")

    target_link_libraries(koliop_fortran_interface
                          # NOTE: I set it to PRIVATE, but this may cause
                          # issues:
                          # https://gitlab.kitware.com/cmake/cmake/-/issues/24402
                          PRIVATE "koliop::koliop")
endif()

########################################
### Dependency handling
########################################

# Call the dependency defined functions on the target.
koliop_target_add_kokkos(koliop)
koliop_target_add_kokkos_kernels(koliop)

koliop_dependent_target_build_options(koliop)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# ########################################
# ### Installation rules
# ########################################

install(TARGETS koliop
        EXPORT koliopTargets
        FILE_SET HEADERS)

install(EXPORT koliopTargets
        NAMESPACE koliop::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/koliop)

configure_package_config_file(cmake/koliopConfig.cmake.in
                              ${CMAKE_CURRENT_BINARY_DIR}/cmake/koliopConfig.cmake
                              INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/koliop)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cmake/koliopConfig.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/koliop)

################################################################################
### Tests
################################################################################

if(koliop_BUILD_TESTING)
    add_subdirectory("tests")
endif()
