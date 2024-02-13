message(STATUS "[koliop] Seeking LTO support.")

option(koliop_ENABLE_LTO "Use Link Time Optimizations (LTO)." ON)

if(koliop_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT koliop_LTO_ENABLED)

    if(koliop_LTO_ENABLED)
        message(STATUS "[koliop]     LTO support found.")
    endif()
endif()

function(koliop_target_enable_lto target_name)
    if(koliop_LTO_ENABLED AND
       CMAKE_BUILD_TYPE MATCHES "Release")
        # NOTE: We have had some issues with LTO and the Clang based compilers
        # on Adastra. Note that recent Clang offer the -foffload-lto, an LTO
        # pass that is device/target aware.
        # NOTE: When using kokkos and a CUDA backend, make sure to use the
        # nvcc_wrapper or else, you will confuse CMake into thinking there is
        # LTO support where as in practice, it is not (you end up linking g++
        # LTOed object files with nvcc non LTOed files.)
        # NOTE: As much as I would like to use LTO, too many compilers just
        # blatantly fail at making it work in production situation.
        set_target_properties("${target_name}" PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endfunction()
