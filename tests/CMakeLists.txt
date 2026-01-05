################################################################################
### Helpers
################################################################################

option(${PROJECT_NAME}_RUN_DEMONSTRATION_AS_TEST "Run the demonstrations as a test (only checking the return value and if it does not crash)." ON)

function(${PROJECT_NAME}_add_test build_options_function kind namespace test_name)
    set(target_name "${kind}_${test_name}")

    add_executable("${target_name}"
                   "${CMAKE_CURRENT_SOURCE_DIR}/${namespace}/${test_name}.cc")

    cmake_language(CALL "${build_options_function}" "${target_name}")

    ########################################
    ### Boiler plate
    ########################################

    # "return" by writing to a variable named ${test_name}
    # Returns, the name of the created target
    set("${test_name}" "${target_name}"  PARENT_SCOPE)

    if (kind STREQUAL "test")
        message(STATUS "[${PROJECT_NAME}] Adding test '${target_name}'.")
        add_test(NAME "${target_name}"
                 # The ctest file is in the build tree's root folder but will call executable inside test
                 # and forwarding it's own working directory to the process. If enable_testing was
                 # in this file instead of the "main" cmake file, the ctest file would be in build/tests
                 # and the tests would be broken because the tests rely on working directory to load files for instance.
                 WORKING_DIRECTORY "." # don't change CWD
                 COMMAND "${target_name}" "in_pipeline")
    elseif(kind STREQUAL "demo")
        message(STATUS "[${PROJECT_NAME}] Adding demonstration '${target_name}'.")
        if(${PROJECT_NAME}_RUN_DEMONSTRATION_AS_TEST)
            add_test(NAME "${target_name}"
                     WORKING_DIRECTORY "."
                     COMMAND "${target_name}" "in_pipeline")
        endif()
    else()
        message(WARNING "Unknown test kind: '${kind}'.")
    endif()
endfunction()


################################################################################
### Test definitions
################################################################################

function(koliop_tests_target_build_options target_name)
    target_link_libraries("${target_name}"
                          PRIVATE "koliop::koliop")

    koliop_dependent_target_build_options("${target_name}")
endfunction()

############################## dummy test ######################################

# Dirty un-scalable check that does the trick.
if(NOT koliop_ASSUME_INPUT_BUFFERS_ARE_DEVICE_COMPATIBLE)
    # The test's memory is initially on the host, we may have to push it to
    # the GPU, thus the restriction.
    message(WARNING "[${PROJECT_NAME}] The 'dummy_integration' test requires koliop_ASSUME_INPUT_BUFFERS_ARE_DEVICE_COMPATIBLE=ON unless unified memory is available.")
endif()

koliop_add_test("koliop_tests_target_build_options" "demo" "dummy" "dummy_integration")
