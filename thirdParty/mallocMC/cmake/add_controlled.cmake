cmake_minimum_required(VERSION 3.20)
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/CPM_0.40.2.cmake)

#  Adds a controlled dependency to the project.
#  Arguments:
#  NAME -- The name of the dependency.
#  Optional Arguments:
#  REQUIRED -- A boolean switch indicating if the dependency is required. Default is OFF.
#  PREFIX -- The prefix for the variable name. Default is the project name.
#  Example:
#  in CMakeLists.txt: add_controlled(NAME my_dependency REQUIRED ON PREFIX my_project)
#  during build: cmake -Dmy_project_USE_my_dependency=ON_ALWAYS_FETCH
function(add_controlled NAME)
  # Parse arguments
  cmake_parse_arguments(ADD_CONTROLLED
    "REQUIRED"  # Boolean options
    "PREFIX"    # Single-value options
    ""          # Multi-value options
    ${ARGN}
  )

  # Set default values if not provided
  if(NOT ADD_CONTROLLED_PREFIX)
    set(ADD_CONTROLLED_PREFIX ${PROJECT_NAME})
  endif()

  if (TARGET ${NAME} OR TARGET ${NAME}::${NAME})
    message("There already exists a target for dependency ${NAME}. Not addding ${NAME} for ${ADD_CONTROLLED_PREFIX} again.")
    return()
  endif()


  set(ALL_OPTIONS "ON;ON_ALLOW_FETCH;ON_ALWAYS_FETCH;AUTO;OFF")
  if(ADD_CONTROLLED_REQUIRED)
    # This is a required dependency, so we're only free to choose how, not if, we want to use it.
    set(AVAILABLE_OPTIONS "ON;ON_ALLOW_FETCH;ON_ALWAYS_FETCH")
    set(${ADD_CONTROLLED_PREFIX}_USE_${NAME} "ON_ALLOW_FETCH" CACHE STRING "")
  else()
    set(AVAILABLE_OPTIONS ${ALL_OPTIONS})
    set(${ADD_CONTROLLED_PREFIX}_USE_${NAME} "AUTO" CACHE STRING "")
  endif()

  if(NOT ${ADD_CONTROLLED_PREFIX}_USE_${NAME} IN_LIST ALL_OPTIONS)
    if(EXISTS ${${ADD_CONTROLLED_PREFIX}_USE_${NAME}})
      set(CPM_${NAME}_SOURCE ${${ADD_CONTROLLED_PREFIX}_USE_${NAME}})
    else()
      message(FATAL_ERROR "You must choose one of ${AVAILABLE_OPTIONS} for ${ADD_CONTROLLED_PREFIX}_USE_${NAME} or a valid path. You've given ${${ADD_CONTROLLED_PREFIX}_USE_${NAME}}.")
    endif()
  elseif(NOT ${ADD_CONTROLLED_PREFIX}_USE_${NAME} IN_LIST AVAILABLE_OPTIONS)
    message(FATAL_ERROR "You must choose one of ${AVAILABLE_OPTIONS} for ${ADD_CONTROLLED_PREFIX}_USE_${NAME} or a valid path. You've given ${${ADD_CONTROLLED_PREFIX}_USE_${NAME}}.")
  endif()

  if (${ADD_CONTROLLED_PREFIX}_USE_${NAME} STREQUAL "OFF")
    return()
  endif()

  # Our default for ON and AUTO:
  set(CPM_USE_LOCAL_PACKAGES ON)
  set(CPM_LOCAL_PACKAGES_ONLY ON)

  if ("${${ADD_CONTROLLED_PREFIX}_USE_${NAME}}" MATCHES "ON_ALLOW_FETCH")
    set(CPM_USE_LOCAL_PACKAGES ON)
    set(CPM_LOCAL_PACKAGES_ONLY OFF)
  elseif ("${${ADD_CONTROLLED_PREFIX}_USE_${NAME}}" MATCHES "ON_ALWAYS_FETCH")
    set(CPM_USE_LOCAL_PACKAGES OFF)
    set(CPM_LOCAL_PACKAGES_ONLY OFF)
  endif()

  # all the details about version, url, etc. are given in cmake/package-lock.cmake
  if ("${${ADD_CONTROLLED_PREFIX}_USE_${NAME}}" MATCHES "^ON")
    CPMAddPackage(NAME ${NAME} REQUIRED)
  else()
    CPMAddPackage(NAME ${NAME})
  endif()
endfunction()
