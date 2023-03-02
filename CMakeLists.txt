cmake_minimum_required(VERSION 3.16)

project("koliop" VERSION 0.0.1 LANGUAGES CXX)

include(CheckIPOSupported)
include(FetchContent)
include(CTest)

################################################################################
### Global options. Dependency specific options are defined in the dependencies.
################################################################################

option(MYLIB_ASSERT_ENABLED "Enable assertions. This enable/disable KOLIOP_ASSERT() but not C's assert()." ON)


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

message(STATUS "Build type is '${CMAKE_BUILD_TYPE}', configuration types are '${CMAKE_CONFIGURATION_TYPES}'.")
message(STATUS "Assuming the compiler is ${CMAKE_CXX_COMPILER_ID}.")

function(koliop_global_target_build_options target_name)
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
    mylib_target_try_enable_lto("${target_name}")
endfunction()




################################################################################
### MYLIB library
################################################################################

add_library(mylib
            STATIC)

set(mylib_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sources")
set(mylib_HEADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/MYLIB/internal/buildspecificities.h.in"
               "${CMAKE_CURRENT_SOURCE_DIR}/include/MYLIB/internal/buildspecificities.h")

# NOTE: GLOBing file is not considered good practice. One could instead specify
# the full, verbose lsit of ehader and source.
file(GLOB_RECURSE mylib_SOURCES "${mylib_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE mylib_HEADERS "${mylib_HEADER_DIR}/*.h")

target_sources(mylib
               PRIVATE ${mylib_SOURCES}
                       ${mylib_HEADERS})

target_include_directories(mylib
                           PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                  $<INSTALL_INTERFACE:include>)

########################################
### Compilation and linking
########################################

mylib_target_add_threads(mylib)
mylib_target_try_add_openmp(mylib)
mylib_target_try_add_mpi(mylib)
mylib_target_try_add_hip(mylib)
mylib_target_add_kokkos(mylib)

mylib_target_compile_options(mylib)


################################################################################
### Tests
################################################################################

# BUILD_TESTING is an implicit option added by CTest.
if(BUILD_TESTING)
    add_subdirectory("tests")
endif()

# TODO(Etienne M): installation
