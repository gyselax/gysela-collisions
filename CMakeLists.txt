cmake_minimum_required(VERSION 3.20)

project("koliop" VERSION 0.0.15 LANGUAGES CXX)

include(CTest)

if(koliop_TOOLCHAIN_NAME)
    message(STATUS "[koliop] Using Koliop toolchain: '${koliop_TOOLCHAIN_NAME}'.")
endif()


################################################################################
### Global options. Dependency specific options are defined in the dependencies.
################################################################################

option(koliop_ASSERT_ENABLED "Enable assertions. This enable/disable KOLIOP_ASSERT() but not C's assert()." ON)
option(koliop_FENCE_ON_OPERATOR_EXIT "Fences before returning from an operator method." OFF)
option(koliop_ASSUME_INPUT_BUFFERS_ARE_DEVICE_COMPATIBLE "Should koliop assumes the input data can be read or else, intermediate copies are required." ON)
option(koliop_BUILD_FORTRAN_INTERFACE "Enable the koliop::fortran_interface glue code needed to interface koliop from Fortran." OFF)

################################################################################
### Dependencies
################################################################################

include(${CMAKE_CURRENT_LIST_DIR}/dependencies/resolver.cmake)

DependencyResolver("${CMAKE_CURRENT_SOURCE_DIR}/dependencies")


################################################################################
### Compiler flags
################################################################################

message(STATUS "[koliop] Build type is '${CMAKE_BUILD_TYPE}', configuration types are '${CMAKE_CONFIGURATION_TYPES}'.")
message(STATUS "[koliop] Assuming the compiler is '${CMAKE_CXX_COMPILER_ID}'.")

# Basis used by the tests. Does not link with private (internal) interfaces like
# Kokkos.
function(koliop_dependent_target_build_options target_name)
    # Use the flags defined in the toolchain. If the
    # koliop_[FLAGS|LINKER_FLAGS]_* are not defined, they are ignored transparently.
    set(koliop_FLAGS                               ${koliop_FLAGS_BASE}
                                 $<$<CONFIG:DEBUG>:${koliop_FLAGS_DEBUG}>
                            $<$<CONFIG:MINSIZEREL>:${koliop_FLAGS_MINSIZEREL}>
                               $<$<CONFIG:RELEASE>:${koliop_FLAGS_RELEASE}>
                        $<$<CONFIG:RELWITHDEBINFO>:${koliop_FLAGS_RELWITHDEBINFO}>)

    set(koliop_LINKER_FLAGS              $<$<CONFIG:DEBUG>:${koliop_LINKER_FLAGS_DEBUG}>
                                    $<$<CONFIG:MINSIZEREL>:${koliop_LINKER_FLAGS_MINSIZEREL}>
                                       $<$<CONFIG:RELEASE>:${koliop_LINKER_FLAGS_RELEASE}>
                                $<$<CONFIG:RELWITHDEBINFO>:${koliop_LINKER_FLAGS_RELWITHDEBINFO}>)

    target_compile_options("${target_name}"
                           PRIVATE ${koliop_FLAGS})

    target_link_options("${target_name}"
                        PRIVATE ${koliop_LINKER_FLAGS})

    # Call the dependency defined functions on the target.
    koliop_target_enable_lto("${target_name}")
endfunction()

# Koliop specific build options for private interface, say, Kokkos.
function(koliop_target_build_options target_name)
    koliop_dependent_target_build_options("${target_name}")

    # Call the dependency defined functions on the target.
    koliop_target_add_kokkos("${target_name}")
    koliop_target_add_kokkos_kernels("${target_name}")
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
               "${koliop_INCLUDE_DIRECTORY}/KOLIOP/internal/buildspecificities.h")

if(koliop_ASSERT_ENABLED)
    target_compile_definitions(koliop PRIVATE KOLIOP_ASSERT_ENABLED)
endif()

if(koliop_FENCE_ON_OPERATOR_EXIT)
    target_compile_definitions(koliop PRIVATE KOLIOP_FENCE_ON_OPERATOR_EXIT)
endif()

if(koliop_ASSUME_INPUT_BUFFERS_ARE_DEVICE_COMPATIBLE)
    target_compile_definitions(koliop PRIVATE KOLIOP_ASSUME_INPUT_BUFFERS_ARE_DEVICE_COMPATIBLE)
endif()

# NOTE: GLOBing file is not considered good practice. One could instead specify
# the full, verbose list of headers and sources.
file(GLOB_RECURSE koliop_SOURCES_GLOB "${koliop_SOURCES_DIRECTORY}/*.cc")

target_sources(koliop
               PRIVATE ${koliop_SOURCES_GLOB})

target_include_directories(koliop
                           PRIVATE "${koliop_SOURCES_DIRECTORY}/")

target_include_directories(koliop
                           PUBLIC "$<BUILD_INTERFACE:${koliop_INCLUDE_DIRECTORY}/>"
                                  "$<INSTALL_INTERFACE:include/>")

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
                          PRIVATE koliop)
endif()

########################################
### Compilation and linking
########################################

koliop_target_build_options(koliop)


################################################################################
### Tests
################################################################################

# BUILD_TESTING is an implicit option added by CTest.
if(BUILD_TESTING)
    add_subdirectory("tests")
endif()

# TODO(Etienne M): installation
