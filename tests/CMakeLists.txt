################################################################################
### Helpers
################################################################################

function(koliop_add_test executable_subject_path executable_subject)
    set(target_name "test_${executable_subject}")

    add_executable("${target_name}"
                   "${CMAKE_CURRENT_SOURCE_DIR}/${executable_subject_path}/${executable_subject}.cpp")

    target_link_libraries("${target_name}"
                          PRIVATE koliop)

    koliop_dependent_target_build_options("${target_name}")


    ########################################
    ### Boiler plate
    ########################################

    # "return" by writing to a variable named ${executable_subject}
    # Returns, the name of the created target
    set("${executable_subject}" "${target_name}"  PARENT_SCOPE)

    message(STATUS "Adding test ${target_name}")
    add_test(NAME "${target_name}"
             WORKING_DIRECTORY "." # don't change CWD
             COMMAND "${target_name}" "in_pipeline")
endfunction()


################################################################################
### Definitions
################################################################################

######################## dummy test ######################################

koliop_add_test("dummy" "dummy_test")
