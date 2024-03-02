cmake_minimum_required(VERSION 3.25)

include(FetchContent)

# print_target_properties was snatched from
# https://stackoverflow.com/questions/32183975/how-to-print-all-the-properties-of-a-target-in-cmake

# if(NOT _PROPERTY_LIST)
#     execute_process(COMMAND cmake --help-property-list OUTPUT_VARIABLE _PROPERTY_LIST)

#     # Convert command output into a CMake list
#     string(REGEX REPLACE ";" "\\\\;" _PROPERTY_LIST "${_PROPERTY_LIST}")
#     string(REGEX REPLACE "\n" ";" _PROPERTY_LIST "${_PROPERTY_LIST}")
#     list(REMOVE_DUPLICATES _PROPERTY_LIST)
# endif()

# function(print_properties)
#     message("_PROPERTY_LIST = ${_PROPERTY_LIST}")
# endfunction()

# function(print_target_properties target)
#     if(NOT TARGET ${target})
#       message(STATUS "There is no target named '${target}'")
#       return()
#     endif()

#     foreach(property ${_PROPERTY_LIST})
#         string(REPLACE "<CONFIG>" "${CMAKE_BUILD_TYPE}" property ${property})

#         if(property STREQUAL "LOCATION" OR property MATCHES "^LOCATION_" OR property MATCHES "_LOCATION$")
#             continue()
#         endif()

#         get_property(was_set TARGET ${target} PROPERTY ${property} SET)
#         if(was_set)
#             get_target_property(value ${target} ${property})
#             message("${target} ${property} = ${value}")
#         endif()
#     endforeach()
# endfunction()

# Provider kind.
# Provider     | Source | Use case
# OFF          |        | To hell with this dependency, do not ask to resolve the dependency anymore.
# SYSTEM       |        | Check if the dependency is already in the project's context, or if the system provides it. This does not work well with SUBDIRECTORY.
# SUBDIRECTORY |        | Assumes the targets are ALREADY in the context, due to, say, a add_subdirectory. (This is considered bad practice).
# GIT          | URL    | Online git
# ARCHIVE      | URI    | Disk, say a targz or zip
# ARCHIVE      | URL    | Online, say a targz or zip
# ARCHIVE      | URI    | Sub directory containing a CMakeLists.txt
# ARCHIVE      | URI    | External directory containing a CMakeLists.txt
#
# TODO(Etienne M): What happens if a find_package or add_subdirectory satisfying
# a dependency is called before FetchContent_* is still unclear.
function(DependencyResolver_Options _NAME)
    cmake_parse_arguments(PARSE_ARGV 1 "" "" "" "PROVIDERS;PROVIDERS_SOURCE")

    set(${PROJECT_NAME}_ENABLE_${_NAME} "${_PROVIDERS}"        CACHE STRING "Define a provider list for this dependency [OFF|SYSTEM|GIT|ARCHIVE]*. If in a parent project, a dependency is already pulled through whatever means, you can use SYSTEM to use it.")
    set(${PROJECT_NAME}_SOURCE_${_NAME} "${_PROVIDERS_SOURCE}" CACHE STRING "If '${PROJECT_NAME}_ENABLE_${_NAME}' is set to [GIT|ARCHIVE], specify where to locate the resource. ARCHIVE mode accepts an URL or filesystem path towards a archive (say, a tarball) or a directory to sources (external or project subdirectory).")
endfunction()

function(DependencyResolver_ResolveLibrary _NAME)
    cmake_parse_arguments(PARSE_ARGV 1 "" "" "VERSION" "COMPONENTS;OPTIONAL_COMPONENTS;PROVIDERS;PROVIDERS_SOURCE")

    list(LENGTH _PROVIDERS PROVIDER_COUNT)

    while(PROVIDER_COUNT)
        list(POP_FRONT _PROVIDERS A_PROVIDER)

        if("${A_PROVIDER}" STREQUAL "OFF")
            message(STATUS "[${PROJECT_NAME}] Dependency OFF: '${_NAME}'.")

            foreach(A_COMPONENT IN LISTS _COMPONENTS _OPTIONAL_COMPONENTS)
                # Safeguard: if you asked for no dependency but we still got
                # something, potentially from a parent/prior dependency.
                if(TARGET "${A_COMPONENT}")
                    message(FATAL_ERROR "[${PROJECT_NAME}] Found '${A_COMPONENT}' for supposedly deactivated '${_NAME}'. Conflicting signals are signs for problems. Either activate the dependency for project '${PROJECT_NAME}' or understand why '${A_COMPONENT}' is defined. If it is expected that someone else provide '${A_COMPONENT}', you may want to use a SYSTEM dependency in the '${PROJECT_NAME}' project.")
                endif()
            endforeach()

            # If we end up failing, ensure the dependency is marked as OFF
            # entirely instead of a potential list of provider. This way we have
            # clean disabled dependency handling for codes relying on
            # `if(${${PROJECT_NAME}_ENABLE_${_NAME}})` to know if they can
            # assume the dependency is resolved (say for a
            # target_link_libraries).

            set(${PROJECT_NAME}_ENABLE_${_NAME} "OFF" CACHE STRING "" FORCE)

            return()
        elseif("${A_PROVIDER}" STREQUAL "SUBDIRECTORY")
            message(WARNING "[${PROJECT_NAME}] Dependency SUBDIRECTORY: '${_NAME}'.")

            # NOTE: Assume all the components are already in context.

            foreach(A_COMPONENT IN LISTS _COMPONENTS)
                # Ensure all required components are resolved.
                if(NOT TARGET "${A_COMPONENT}")
                    message(FATAL_ERROR "[${PROJECT_NAME}] You specified that you rely on a subdirectory to expose the '${_NAME}' component but it was not found.")
                endif()
            endforeach()

            # Unfortunately, if we fail to find our dependencies we can't
            # continue because we have no way of knowing what was imported
            # through add_subdirectory and can't rollback the add_subdirectory.
            # We can only completely fail or succeed.
            return()
        elseif("${A_PROVIDER}" MATCHES "SYSTEM|_MakeAvailable_SYSTEM_QUIET")
            if("${A_PROVIDER}" STREQUAL "SYSTEM")
                message(STATUS "[${PROJECT_NAME}] Dependency SYSTEM: '${_NAME}'.")
            endif()

            list(APPEND _OPTIONAL_COMPONENTS "${_COMPONENTS}")

            find_package("${_NAME}" "${_VERSION}" OPTIONAL_COMPONENTS "${_OPTIONAL_COMPONENTS}")

            if("${${_NAME}_FOUND}")
                foreach(A_COMPONENT IN LISTS _COMPONENTS)
                    # print_target_properties("${A_COMPONENT}")

                    # NOTE: We use OPTIONAL_COMPONENTS to avoid the hard error if
                    # the package is not found. But this mean we need to ensure all
                    # needed components are presents.
                    if(NOT TARGET "${A_COMPONENT}")
                        # NOTE: We found the dependency but failed to satisfy
                        # all required components. We have to hard error because
                        # the state of the environment changed
                        # ("${${_NAME}_FOUND}" is set). To continue we
                        # would have to rollback the state. This is significant
                        # issue IMO.
                        if("${A_PROVIDER}" STREQUAL "SYSTEM")
                            message(FATAL_ERROR "[${PROJECT_NAME}] Found '${_NAME}' but at least '${A_COMPONENT}' was not satisfied. Either fix the SYSTEM installation or reconfigure without the SYSTEM provider.")
                        else()
                            # We came from the fallthrough of GIT or ARCHIVE.
                            message(FATAL_ERROR "[${PROJECT_NAME}] Found '${_NAME}' but at least '${A_COMPONENT}' was not satisfied. It is possible the dependency was not properly configured, we are expecting components it does not currently provides.")
                        endif()
                    endif()
                endforeach()

                # If we get here, we found what we need, else we would have
                # failed earlier.

                return()
            elseif("${A_PROVIDER}" STREQUAL "_MakeAvailable_SYSTEM_QUIET")
                # Not found but we came from an other provider which used
                # FetchContent. We should never get here as
                # FetchContent_MakeAvailable never fails to provide something,
                # it crashes if it cant.
                message(FATAL_ERROR "[${PROJECT_NAME}] Conflicting signals, we did not found '${_NAME}' but FetchContent_MakeAvailable said it did.")
            endif()

            # Fallthrough and continue.
        elseif("${A_PROVIDER}" STREQUAL "GIT")
            message(STATUS "[${PROJECT_NAME}] Dependency GIT: '${_NAME}'.")

            # Each source is matched with its provider (SOA fashion).
            list(POP_FRONT _PROVIDERS_SOURCE A_PROVIDER_SOURCE)

            FetchContent_Declare("${_NAME}"
                                 SYSTEM
                                 OVERRIDE_FIND_PACKAGE
                                 # TODO(Etienne M): We would benefit from a
                                 # GIT_TAG
                                 GIT_REPOSITORY        "${A_PROVIDER_SOURCE}"
                                 # These timeouts are obviously no honored..
                                 TIMEOUT               10
                                 INACTIVITY_TIMEOUT    10
                                 )

            FetchContent_MakeAvailable("${_NAME}")

            # TODO(Etienne M): If one day we get a way to check if
            # FetchContent_MakeAvailable fails or succeeds, change this
            # branch.
            # https://discourse.cmake.org/t/prevent-fatal-error-on-fetchcontent-declare/10091/2
            if(TRUE)
                # NOTE: FetchContent_MakeAvailable alone does not define
                # ${${_NAME}_FOUND}. We used OVERRIDE_FIND_PACKAGE so find_package
                # is rerouted to FetchContent_MakeAvailable and basically does
                # nothing more than defining ${${_NAME}_FOUND}. From the user
                # perspective, it really looks as if it was found on the system.
                # We could use something like the line below or just PREPEND a
                # _MakeAvailable_SYSTEM_QUIET.
                # find_package("${_NAME}" "${_VERSION}" OPTIONAL_COMPONENTS "${_OPTIONAL_COMPONENTS}")
                list(PREPEND _PROVIDERS "_MakeAvailable_SYSTEM_QUIET")
                # Fallthrough and continue to _MakeAvailable_SYSTEM_QUIET.
            else()
                # Fallthrough and continue.
            endif()
        elseif("${A_PROVIDER}" STREQUAL "ARCHIVE")
            message(STATUS "[${PROJECT_NAME}] Dependency ARCHIVE: '${_NAME}'.")

            # Each source is matched with its provider (SOA fashion).
            list(POP_FRONT _PROVIDERS_SOURCE A_PROVIDER_SOURCE)

            # There is no simple CMake way to check if A_PROVIDER_SOURCE
            # represents a path. This is how it is done in
            # ExternalProject.cmake.
            if(NOT "${A_PROVIDER_SOURCE}" MATCHES "^[a-z]+://")
                # We are probably dealing with a file path or directory.
                if(NOT EXISTS "${A_PROVIDER_SOURCE}")
                    # Special warning because FetchContent does not send the correct
                    # signals. It says it expect archives only which is wrong, it
                    # can take directories too.
                    message(WARNING "[${PROJECT_NAME}] Could not find directory or file '${A_PROVIDER_SOURCE}'.")
                    continue()
                endif()
            endif()

            FetchContent_Declare("${_NAME}"
                                 SYSTEM
                                 OVERRIDE_FIND_PACKAGE
                                 # TODO(Etienne M): We would benefit from a
                                 # URL_HASH
                                 # A URL may be an ordinary path in the local
                                 # file system (in which case it must be the
                                 # only URL provided) or any downloadable URL
                                 # supported by the file(DOWNLOAD) command.
                                 URL                   "${A_PROVIDER_SOURCE}"
                                 # These timeouts are obviously no honored..
                                 TIMEOUT               10
                                 INACTIVITY_TIMEOUT    10
                                 )

            FetchContent_MakeAvailable("${_NAME}")

            # TODO(Etienne M): If one day we get a way to check if
            # FetchContent_MakeAvailable fails or succeeds, change this
            # branch.
            # https://discourse.cmake.org/t/prevent-fatal-error-on-fetchcontent-declare/10091/2
            if(TRUE)
                list(PREPEND _PROVIDERS "_MakeAvailable_SYSTEM_QUIET")
                # Fallthrough and continue to _MakeAvailable_SYSTEM_QUIET.
            else()
                # Fallthrough and continue.
            endif()
        else()
            message(FATAL_ERROR "[${PROJECT_NAME}] Unknown provider '${A_PROVIDER}'.")
        endif()

        # This needs to be at the end because we expect some provider to modify
        # the queue length.
        list(LENGTH _PROVIDERS PROVIDER_COUNT)
    endwhile()

    message(FATAL_ERROR "[${PROJECT_NAME}] Dependency **not found**: '${_NAME}'.")
endfunction()

# Example:
# DependencyResolver_Resolve(Kokkos
#                            VERSION          "4.1.0"
#                            COMPONENTS       Kokkos::kokkos
#                            PROVIDERS        ARCHIVE
#                                             GIT
#                                             ARCHIVE
#                                             ARCHIVE
#                                             SYSTEM
#                                             OFF
#                            PROVIDERS_SOURCE "../kokkos"
#                                             "https://github.com/kokkos/kokkos"
#                                             "https://github.com/kokkos/kokkos/archive/refs/tags/4.2.00.tar.gz"
#                                             "../4.2.00.tar.gz"
#                            )
function(DependencyResolver_Resolve _NAME)
    cmake_parse_arguments(PARSE_ARGV 1 "" "" "VERSION" "COMPONENTS;OPTIONAL_COMPONENTS;PROVIDERS;PROVIDERS_SOURCE")

    # Define default options (cached variables).
    DependencyResolver_Options("${_NAME}"
                                    PROVIDERS        ${_PROVIDERS}
                                    PROVIDERS_SOURCE ${_PROVIDERS_SOURCE}
                                    )
    # Now, forward the cached variables, not the one given as argument to
    # DependencyResolver_Resolve.
    DependencyResolver_ResolveLibrary("${_NAME}"
                                      VERSION             "${_VERSION}"
                                      COMPONENTS          ${_COMPONENTS}
                                      OPTIONAL_COMPONENTS ${_OPTIONAL_COMPONENTS}
                                      PROVIDERS           ${${PROJECT_NAME}_ENABLE_${_NAME}}
                                      PROVIDERS_SOURCE    ${${PROJECT_NAME}_SOURCE_${_NAME}}
                                      )
endfunction()

# The needs to be a macro to propagate upward the variable defined by the
# dependencies.
macro(DependencyResolver _PATH)
    file(GLOB THE_DEPENDENCIES "${_PATH}/*")

    foreach(A_DEPENDENCY ${THE_DEPENDENCIES})
        if(IS_DIRECTORY "${A_DEPENDENCY}")
            include("${A_DEPENDENCY}/seeker.cmake")
        endif()
    endforeach()
endmacro()
