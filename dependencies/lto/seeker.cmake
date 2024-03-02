option(${PROJECT_NAME}_ENABLE_LTO "Use Link Time Optimizations (LTO)." ON)

if(${PROJECT_NAME}_ENABLE_LTO)
    message(STATUS "[${PROJECT_NAME}] Dependency seeking LTO.")

    include(CheckIPOSupported)
    check_ipo_supported(RESULT ${PROJECT_NAME}_LTO_ENABLED)

    if(${PROJECT_NAME}_LTO_ENABLED)
        message(STATUS "[${PROJECT_NAME}]     Dependency LTO found.")
    endif()
else()
    message(STATUS "[${PROJECT_NAME}] Dependency skipping LTO.")
endif()

function(${PROJECT_NAME}_target_enable_lto target_name)
    if(${PROJECT_NAME}_LTO_ENABLED AND
       CMAKE_BUILD_TYPE MATCHES "Release")
        set_target_properties("${target_name}" PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endfunction()
