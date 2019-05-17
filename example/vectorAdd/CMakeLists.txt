#
# Copyright 2014-2019 Benjamin Worpitz
#
# This file exemplifies usage of Alpaka.
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED “AS IS” AND ISC DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL ISC BE LIABLE FOR ANY
# SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
# IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#

################################################################################
# Required CMake version.

CMAKE_MINIMUM_REQUIRED(VERSION 3.11.0)

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Project.

SET(_TARGET_NAME "vectorAdd")

PROJECT(${_TARGET_NAME})

################################################################################
# CMake policies
#
# Search in <PackageName>_ROOT:
#   https://cmake.org/cmake/help/v3.12/policy/CMP0074.html

if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

#-------------------------------------------------------------------------------
# Find alpaka.

SET(ALPAKA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../" CACHE STRING "The location of the alpaka library")
LIST(APPEND CMAKE_MODULE_PATH "${ALPAKA_ROOT}")
FIND_PACKAGE("alpaka" REQUIRED)

#-------------------------------------------------------------------------------
# Common.

INCLUDE("${ALPAKA_ROOT}cmake/common.cmake")
INCLUDE("${ALPAKA_ROOT}cmake/dev.cmake")

#-------------------------------------------------------------------------------
# Add executable.

SET(_SOURCE_DIR "src/")

# Add all the source files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_src_group("${_SOURCE_DIR}" "${_SOURCE_DIR}" "cpp" _FILES_SOURCE)

SET(_COMPILE_OPTIONS_PRIVATE ${ALPAKA_DEV_COMPILE_OPTIONS})

IF(MSVC)
    LIST(APPEND _COMPILE_OPTIONS_PRIVATE "/wd4996")   # This function or variable may be unsafe. Consider using <safe_version> instead.
ENDIF()

# NVCC does not incorporate the COMPILE_OPTIONS of a target but only the CMAKE_CXX_FLAGS
IF(ALPAKA_ACC_GPU_CUDA_ENABLE AND NOT ALPAKA_CUDA_COMPILER MATCHES "clang")
    STRING(REPLACE ";" " " _COMPILE_OPTIONS_PRIVATE_STRING "${_COMPILE_OPTIONS_PRIVATE}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_COMPILE_OPTIONS_PRIVATE_STRING}")
ENDIF()

# Always add all files to the target executable build call to add them to the build project.
ALPAKA_ADD_EXECUTABLE(
    ${_TARGET_NAME}
    ${_FILES_SOURCE})
TARGET_COMPILE_OPTIONS(
    ${_TARGET_NAME}
    PRIVATE ${_COMPILE_OPTIONS_PRIVATE})
TARGET_LINK_LIBRARIES(
    ${_TARGET_NAME}
    PUBLIC "alpaka")

SET_TARGET_PROPERTIES(${_TARGET_NAME} PROPERTIES FOLDER "example")

ADD_TEST(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME})
