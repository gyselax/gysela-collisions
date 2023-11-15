################################################################################
### Helpers
################################################################################

function(koliop_add_test test_category test_name)
    set(target_name "test_${test_name}")

    message(STATUS "[koliop] Adding test '${target_name}'.")

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

koliop_add_test("dummy" "dummy_test")
