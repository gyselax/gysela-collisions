################################################################################
### Helpers
################################################################################

function(koliop_add_test test_category test_name)
    set(target_name "test_${test_name}")

    message(STATUS "[${PROJECT_NAME}] Adding test '${target_name}'.")

    add_executable("${target_name}"
                   "${CMAKE_CURRENT_SOURCE_DIR}/${test_category}/${test_name}.cc")

    target_link_libraries("${target_name}"
                          PRIVATE "koliop::koliop")

    koliop_dependent_target_build_options("${target_name}")

    add_test(NAME "${target_name}"
             WORKING_DIRECTORY "." # don't change CWD
             COMMAND "${target_name}" "in_pipeline")

    # "return" by writing to a variable named ${test_name}
    # Returns, the name of the created target
    set("${test_name}" "${target_name}"  PARENT_SCOPE)
endfunction()


################################################################################
### Test definition
################################################################################

############################## dummy test ######################################

# Dirty un-scalable check that does the trick.
if(koliop_ASSUME_INPUT_BUFFERS_ARE_DEVICE_COMPATIBLE AND NOT(Kokkos_ENABLE_OPENMP OR Kokkos_ENABLE_SERIAL))
    # The test's memory is initially on the host, we may have to push it to
    # the GPU, thus the restriction.
    message(WARNING "[${PROJECT_NAME}] The 'dummy_test' test requires koliop_ASSUME_INPUT_BUFFERS_ARE_DEVICE_COMPATIBLE=OFF.")
else()
    koliop_add_test("dummy" "dummy_test")
endif()
