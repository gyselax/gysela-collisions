cmake_minimum_required(VERSION 3.16)

project("koliop" VERSION 0.0.1 LANGUAGES CXX)

include(CheckIPOSupported)
include(FetchContent)
include(CTest)

if(koliop_TOOLCHAIN_NAME)
    message(STATUS "[koliop] Using Koliop toolchain: '${koliop_TOOLCHAIN_NAME}'.")
else()
    message(FATAL_ERROR "[koliop] koliop_TOOLCHAIN_NAME is not defined.")
endif()


################################################################################
### Global options. Dependency specific options are defined in the dependencies.
################################################################################

option(KOLIOP_ASSERT_ENABLED "Enable assertions. This enable/disable KOLIOP_ASSERT() but not C's assert()." ON)


################################################################################
### Dependencies
################################################################################

file(GLOB dependencies "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/*")

foreach(a_dependency ${dependencies})
    if(IS_DIRECTORY "${a_dependency}")
        include("${a_dependency}/seeker.cmake")
    endif()
endforeach()


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

set(koliop_SOURCES_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/sources")
set(koliop_INCLUDE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include")

configure_file("${koliop_INCLUDE_DIRECTORY}/KOLIOP/internal/buildspecificities.h.in"
               "${koliop_INCLUDE_DIRECTORY}/KOLIOP/internal/buildspecificities.h")

# NOTE: GLOBing file is not considered good practice. One could instead specify
# the full, verbose list of headers and sources.
file(GLOB_RECURSE koliop_SOURCES_GLOB "${koliop_SOURCES_DIRECTORY}/*.cc")

target_sources(koliop
               PRIVATE ${koliop_SOURCES_GLOB})

target_include_directories(koliop
                           PRIVATE "${koliop_SOURCES_DIRECTORY}/")

target_include_directories(koliop
                           PUBLIC   "$<BUILD_INTERFACE:${koliop_INCLUDE_DIRECTORY}/>"
                                    "$<INSTALL_INTERFACE:include/>")

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
