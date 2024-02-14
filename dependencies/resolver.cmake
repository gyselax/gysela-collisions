cmake_minimum_required(VERSION 3.25)

# Provider kind.
# OFF
# SYSTEM / Already provided
# GIT     (git URL)
# ARCHIVE (disk URI)
# ARCHIVE (online URL)
# ARCHIVE (sub dir URI)
# ARCHIVE (external dir URI)
function(_SPLBDependencyResolver_Options DEPENDENCY_NAME)
    cmake_parse_arguments(PARSE_ARGV 1 DEPENDENCY "" "" "PROVIDERS;PROVIDERS_SOURCE")

    set(${CMAKE_PROJECT_NAME}_ENABLE_${DEPENDENCY_NAME} "${DEPENDENCY_PROVIDERS}"        CACHE STRING "Define a provider list for this dependency [OFF|SYSTEM|GIT|ARCHIVE]*. If in a parent project, a dependency is already pulled through whatever means, you can use SYSTEM to use it.")
    set(${CMAKE_PROJECT_NAME}_SOURCE_${DEPENDENCY_NAME} "${DEPENDENCY_PROVIDERS_SOURCE}" CACHE STRING "If '${CMAKE_PROJECT_NAME}_ENABLE_${DEPENDENCY_NAME}' is set to [GIT|ARCHIVE], specify where to locate the resource. ARCHIVE mode accepts an URL or filesystem path towards a archive (say, a tarball) or a directory to sources (external or project subdirectory).")
endfunction()

function(_SPLBDependencyResolver_ResolveLibrary DEPENDENCY_NAME)
    cmake_parse_arguments(PARSE_ARGV 1 DEPENDENCY "" "VERSION" "COMPONENTS;OPTIONAL_COMPONENTS;PROVIDERS;PROVIDERS_SOURCE")

    list(LENGTH DEPENDENCY_PROVIDERS PROVIDER_COUNT)

    while(PROVIDER_COUNT)
        list(POP_FRONT DEPENDENCY_PROVIDERS A_PROVIDER)
        list(LENGTH DEPENDENCY_PROVIDERS PROVIDER_COUNT)

        if("${A_PROVIDER}" STREQUAL "OFF")
            message(STATUS "[${CMAKE_PROJECT_NAME}] Dependency OFF:       '${DEPENDENCY_NAME}'.")

            foreach(A_COMPONENT IN LISTS DEPENDENCY_COMPONENTS DEPENDENCY_OPTIONAL_COMPONENTS)
                # Safeguard: if you asked fr no deps but still got something,
                # potentially from a parent/prior dependency.
                if(TARGET "${A_COMPONENT}")
                    message(FATAL_ERROR "[${CMAKE_PROJECT_NAME}] Deactivating '${DEPENDENCY_NAME}' found '${A_COMPONENT}'. Conflicting signals are signs for problems. Either activate the dependency for project '${CMAKE_PROJECT_NAME}' or understand why '${A_COMPONENT}' is defined. If this is expected that someone else provide '${A_COMPONENT}', you may want to use a SYSTEM dependency in the '${CMAKE_PROJECT_NAME}' project.")
                endif()
            endforeach()

            return()
        elseif("${A_PROVIDER}" STREQUAL "SYSTEM")
            message(STATUS "[${CMAKE_PROJECT_NAME}] Dependency SYSTEM:    '${DEPENDENCY_NAME}'.")

            list(APPEND DEPENDENCY_OPTIONAL_COMPONENTS "${DEPENDENCY_COMPONENTS}")

            find_package("${DEPENDENCY_NAME}" "${DEPENDENCY_VERSION}" OPTIONAL_COMPONENTS "${DEPENDENCY_OPTIONAL_COMPONENTS}")

            if("${${DEPENDENCY_NAME}_FOUND}")
                foreach(A_COMPONENT IN LISTS DEPENDENCY_COMPONENTS)
                    # NOTE: We use OPTIONAL_COMPONENTS to avoid the hard error if
                    # the package is not found. But this mean we need to ensure all
                    # needed components are presents.
                    if(NOT TARGET "${A_COMPONENT}")
                        message(FATAL_ERROR "[${CMAKE_PROJECT_NAME}] Found '${DEPENDENCY_NAME}' but not all '${DEPENDENCY_COMPONENTS}' components are satisfied. Either fix the SYSTEM package or reconfigure without the SYSTEM provider.")
                    endif()
                endforeach()

                return()
            endif()
        elseif("${A_PROVIDER}" STREQUAL "GIT")
            message(STATUS "[${CMAKE_PROJECT_NAME}] Dependency GIT: '${DEPENDENCY_NAME}'.")

            # Each source is matched with its provider (SOA fashion).
            list(POP_FRONT DEPENDENCY_PROVIDERS_SOURCE A_PROVIDER_SOURCE)

            FetchContent_Declare("${DEPENDENCY_NAME}"
                                 SYSTEM
                                 OVERRIDE_FIND_PACKAGE
                                 # TODO(Etienne M): We would benefit from a
                                 # GIT_TAG
                                 GIT_REPOSITORY        "${A_PROVIDER_SOURCE}"
                                 # These timeouts are obviously no honored..
                                 TIMEOUT               10
                                 INACTIVITY_TIMEOUT    10
                                 )

            FetchContent_MakeAvailable("${DEPENDENCY_NAME}")

            # NOTE: It is implied that we found a library.

            # TODO(Etienne M): If it fails (timeout, fallback to the next).

            return()
        elseif("${A_PROVIDER}" STREQUAL "ARCHIVE")
            message(STATUS "[${CMAKE_PROJECT_NAME}] Dependency ARCHIVE: '${DEPENDENCY_NAME}'.")

            # Each source is matched with its provider (SOA fashion).
            list(POP_FRONT DEPENDENCY_PROVIDERS_SOURCE A_PROVIDER_SOURCE)

            # There is no simple CMake way to check if A_PROVIDER_SOURCE
            # represents a path.
            if(NOT "${A_PROVIDER_SOURCE}" MATCHES "https?:\/\/(www\.)?.*")
                # We are probably dealing with a file system path.
                if(NOT EXISTS "${A_PROVIDER_SOURCE}")
                    # Special warning because FetchContent does not send the correct
                    # signals. It says it expect archives only which is wrong, it
                    # can take directories too.
                    message(FATAL_ERROR "[${CMAKE_PROJECT_NAME}] Could not find directory or file '${A_PROVIDER_SOURCE}'.")
                endif()
            endif()

            FetchContent_Declare("${DEPENDENCY_NAME}"
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
            FetchContent_MakeAvailable("${DEPENDENCY_NAME}")

            # NOTE: It is implied that we found a library.

            # TODO(Etienne M): If it fails (timeout, fallback to the next).

            return()
        else()
            message(FATAL_ERROR "[${CMAKE_PROJECT_NAME}] Unknown provider '${A_PROVIDER}'.")
        endif()
    endwhile()

    message(FATAL_ERROR "[${CMAKE_PROJECT_NAME}] Dependency **not found**: '${DEPENDENCY_NAME}'.")
endfunction()

# Example:
# _SPLBDependencyResolver_Resolve(Kokkos
# VERSION          "4.1.0"
# COMPONENTS       Kokkos::kokkos
# PROVIDERS        ARCHIVE
#                  GIT
#                  ARCHIVE
#                  ARCHIVE
#                  SYSTEM
#                  OFF
# PROVIDERS_SOURCE "/lus/home/BCINES/dci/malaboeuf/repository.backedup/gysela-gpu-porting/koliop/build/kokkos"
#                  "https://github.com/kokkos/kokkos"
#                  "https://github.com/kokkos/kokkos/archive/refs/tags/4.2.00.tar.gz"
#                  "/lus/home/BCINES/dci/malaboeuf/repository.backedup/gysela-gpu-porting/koliop/4.2.00.tar.gz"
# )
function(_SPLBDependencyResolver_Resolve DEPENDENCY_NAME)
    cmake_parse_arguments(PARSE_ARGV 1 DEPENDENCY "" "VERSION" "COMPONENTS;OPTIONAL_COMPONENTS;PROVIDERS;PROVIDERS_SOURCE")

    # Define default options (cached variables).
    _SPLBDependencyResolver_Options("${DEPENDENCY_NAME}"
                                    PROVIDERS        ${DEPENDENCY_PROVIDERS}
                                    PROVIDERS_SOURCE ${DEPENDENCY_PROVIDERS_SOURCE}
                                    )
    # Now, forward the cached variables, not the one given as argument to
    # _SPLBDependencyResolver_Resolve.
    _SPLBDependencyResolver_ResolveLibrary("${DEPENDENCY_NAME}"
                                           VERSION             "${DEPENDENCY_VERSION}"

                                           COMPONENTS          ${DEPENDENCY_COMPONENTS}
                                           OPTIONAL_COMPONENTS ${DEPENDENCY_OPTIONAL_COMPONENTS}
                                           PROVIDERS           ${${CMAKE_PROJECT_NAME}_ENABLE_${DEPENDENCY_NAME}}
                                           PROVIDERS_SOURCE    ${${CMAKE_PROJECT_NAME}_SOURCE_${DEPENDENCY_NAME}}
                                           )
endfunction()

function(SPLBDependencyResolver DEPENDENCY_PATH)
    file(GLOB THE_DEPENDENCIES "${DEPENDENCY_PATH}/*")

    foreach(A_DEPENDENCY ${THE_DEPENDENCIES})
        if(IS_DIRECTORY "${A_DEPENDENCY}")
            include("${A_DEPENDENCY}/seeker.cmake")
        endif()
    endforeach()
endfunction()
