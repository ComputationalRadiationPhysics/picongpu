#
# Copyright 2014-2017 Benjamin Worpitz
#
# This file is part of alpaka.
#
# alpaka is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# alpaka is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with alpaka.
# If not, see <http://www.gnu.org/licenses/>.
#

################################################################################
# Required CMake version.

CMAKE_MINIMUM_REQUIRED(VERSION 3.7.0)

################################################################################
# alpaka test common.

# Return values.
UNSET(common_FOUND)
UNSET(common_DEFINITIONS)
UNSET(common_INCLUDE_DIR)
UNSET(common_INCLUDE_DIRS)
UNSET(common_LIBRARIES)

# Internal usage.
UNSET(_COMMON_FOUND)
UNSET(_COMMON_INCLUDE_DIRECTORY)
UNSET(_COMMON_INCLUDE_DIRECTORIES_PUBLIC)
UNSET(_COMMON_COMPILE_DEFINITIONS_PUBLIC)
UNSET(_COMMON_COMPILE_OPTIONS_PUBLIC)
UNSET(_COMMON_LINK_LIBRARIES_PUBLIC)
UNSET(_COMMON_SOURCE_DIRECTORY)
UNSET(_COMMON_TARGET_NAME)
UNSET(_COMMON_FILES_HEADER)
UNSET(_COMMON_FILES_SOURCE)
UNSET(_COMMON_FILES_CMAKE)

# Directory of this file.
SET(_COMMON_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})
# Normalize the path (e.g. remove ../)
GET_FILENAME_COMPONENT(_COMMON_ROOT_DIR "${_COMMON_ROOT_DIR}" ABSOLUTE)

# Set found to true initially and set it to false if a required dependency is missing.
SET(_COMMON_FOUND TRUE)

SET(_COMMON_TARGET_NAME "common")

PROJECT(${_COMMON_TARGET_NAME})

#-------------------------------------------------------------------------------
# Find alpaka.

SET(ALPAKA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../" CACHE STRING "The location of the alpaka library")
LIST(APPEND CMAKE_MODULE_PATH "${ALPAKA_ROOT}")
FIND_PACKAGE(alpaka REQUIRED)

IF(NOT alpaka_FOUND)
    MESSAGE(WARNING "Required alpaka test common dependency alpaka could not be found!")
    SET(_COMMON_FOUND FALSE)

ELSE()
    INCLUDE("${ALPAKA_ROOT}cmake/dev.cmake")
    LIST(APPEND _COMMON_COMPILE_OPTIONS_PUBLIC ${ALPAKA_DEV_COMPILE_OPTIONS})
ENDIF()

IF(ALPAKA_ACC_GPU_CUDA_ENABLE)
    LIST(APPEND _COMMON_LINK_LIBRARIES_PUBLIC "general;${CUDA_CUDA_LIBRARY}")
    LIST(APPEND _COMMON_COMPILE_DEFINITIONS_PUBLIC "CUDA_API_PER_THREAD_DEFAULT_STREAM")
ENDIF()

#-------------------------------------------------------------------------------
# Add library.

SET(_COMMON_INCLUDE_DIRECTORY "${_COMMON_ROOT_DIR}/include")
LIST(APPEND _COMMON_INCLUDE_DIRECTORIES_PUBLIC "${_COMMON_INCLUDE_DIRECTORY}")
SET(_COMMON_SOURCE_DIRECTORY "${_COMMON_ROOT_DIR}/src")

# Add all the source files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_src_group("${_COMMON_INCLUDE_DIRECTORY}" "${_COMMON_INCLUDE_DIRECTORY}" "hpp" _COMMON_FILES_HEADER)
append_recursive_files_add_to_src_group("${_COMMON_SOURCE_DIRECTORY}" "${_COMMON_SOURCE_DIRECTORY}" "cpp" _COMMON_FILES_SOURCE)
LIST(APPEND _COMMON_FILES_CMAKE "${_COMMON_ROOT_DIR}/commonConfig.cmake" "${_COMMON_ROOT_DIR}/Findcommon.cmake")

IF(MSVC)
    LIST(APPEND _COMMON_COMPILE_OPTIONS_PUBLIC "/bigobj")
    LIST(APPEND _COMMON_COMPILE_OPTIONS_PUBLIC "/wd4996")   # This function or variable may be unsafe. Consider using <safe_version> instead.
ENDIF()


#-------------------------------------------------------------------------------
# Target.
IF(NOT TARGET ${_COMMON_TARGET_NAME})
    # Always add all files to the target executable build call to add them to the build project.
    ADD_LIBRARY(
        ${_COMMON_TARGET_NAME}
        STATIC
        ${_COMMON_FILES_HEADER} ${_COMMON_FILES_SOURCE} ${_COMMON_FILES_CMAKE})
    # Set the link libraries for this library (adds libs, include directories, defines and compile options).
    TARGET_INCLUDE_DIRECTORIES(
        ${_COMMON_TARGET_NAME}
        PUBLIC ${_COMMON_INCLUDE_DIRECTORIES_PUBLIC})
    LIST(
        LENGTH
        _COMMON_COMPILE_DEFINITIONS_PUBLIC
        _COMMON_COMPILE_DEFINITIONS_PUBLIC_LENGTH)
    IF(${_COMMON_COMPILE_DEFINITIONS_PUBLIC_LENGTH} GREATER 0)
        TARGET_COMPILE_DEFINITIONS(
            ${_COMMON_TARGET_NAME}
            PUBLIC ${_COMMON_COMPILE_DEFINITIONS_PUBLIC})
    ENDIF()
    TARGET_COMPILE_OPTIONS(
        ${_COMMON_TARGET_NAME}
        PUBLIC ${_COMMON_COMPILE_OPTIONS_PUBLIC})
    TARGET_LINK_LIBRARIES(
        ${_COMMON_TARGET_NAME}
        PUBLIC alpaka ${_COMMON_LINK_LIBRARIES_PUBLIC})
    SET_TARGET_PROPERTIES(
        ${_COMMON_TARGET_NAME}
        PROPERTIES FOLDER "test")
ENDIF()

# Unset already set variables if not found.
IF(_COMMON_FOUND)
    # Handles the REQUIRED, QUIET and version-related arguments for FIND_PACKAGE.
    INCLUDE(FindPackageHandleStandardArgs)
    FIND_PACKAGE_HANDLE_STANDARD_ARGS(
        ${_COMMON_TARGET_NAME}
        FOUND_VAR common_FOUND
        REQUIRED_VARS _COMMON_INCLUDE_DIRECTORY)
ENDIF()

UNSET(_COMMON_FOUND)
UNSET(_COMMON_INCLUDE_DIRECTORY)
UNSET(_COMMON_INCLUDE_DIRECTORIES_PUBLIC)
UNSET(_COMMON_COMPILE_DEFINITIONS_PUBLIC)
UNSET(_COMMON_COMPILE_OPTIONS_PUBLIC)
UNSET(_COMMON_LINK_LIBRARIES_PUBLIC)
UNSET(_COMMON_SOURCE_DIRECTORY)
UNSET(_COMMON_TARGET_NAME)
UNSET(_COMMON_FILES_HEADER)
UNSET(_COMMON_FILES_SOURCE)
UNSET(_COMMON_FILES_CMAKE)
