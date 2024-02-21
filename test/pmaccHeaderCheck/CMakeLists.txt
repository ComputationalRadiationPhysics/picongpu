#
# Copyright 2014-2023 Benjamin Worpitz, Axel Huebl, Jan Stephan, Rene Widera
#
# This file is part of PIConGPU.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

cmake_minimum_required(VERSION 3.18)
project(PMaccHeaderTest)

# Add common functions from alpaka.
# Always use the alpaka files shipped with PIConGPU/cupla from the thirdParty directory.
include(${CMAKE_CURRENT_SOURCE_DIR}/../../thirdParty/cupla/alpaka/cmake/common.cmake)

################################################################################
# Directory of this file.
################################################################################
set(PMACC_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../include/pmacc)

# Normalize the path (e.g. remove ../)
get_filename_component(PMACC_ROOT_DIR "${PMACC_ROOT_DIR}" ABSOLUTE)

###############################################################################
# Language Flags
###############################################################################

# enforce C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)

################################################################################
# PMacc
################################################################################
find_package(PMacc REQUIRED CONFIG PATHS "${PMACC_ROOT_DIR}")


set(_TARGET_NAME "headerCheckTest")

###############################################################################
# Catch2
###############################################################################

add_subdirectory(${PMACC_ROOT_DIR}/../../thirdParty/catch2 ${CMAKE_BINARY_DIR}/catch2)

#---------------------------------------------------------------------------
# Create source files.
set(PMACC_SUFFIXED_INCLUDE_DIR "${PMACC_ROOT_DIR}")
append_recursive_files("${PMACC_SUFFIXED_INCLUDE_DIR}" "hpp" "PMACC_FILES_HEADER")
# remove pmacc/test to avoid pulling tests cases where catch is used
list(FILTER PMACC_FILES_HEADER EXCLUDE REGEX ".*pmacc/test.*")

set(_GENERATED_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/src")

file(REMOVE_RECURSE ${_GENERATED_SOURCE_DIR})

foreach (_HEADER_FILE ${PMACC_FILES_HEADER})
    # Remove the parent directory from the path.
    # NOTE: This is not correct because it does not only replace at the beginning of the string.
    #  "STRING(REGEX REPLACE" would be correct if there was an easy way to escape arbitrary strings.
    string(
            REPLACE "${PMACC_SUFFIXED_INCLUDE_DIR}/" ""
            _HEADER_FILE
            "${_HEADER_FILE}")
    set(_SOURCE_FILE "${_GENERATED_SOURCE_DIR}/${_HEADER_FILE}.cpp")
    file(WRITE "${_SOURCE_FILE}" "#include <pmacc/${_HEADER_FILE}>\n#include <pmacc/Environment.hpp>\n")
endforeach ()

#---------------------------------------------------------------------------
# Add executable.

append_recursive_files_add_to_src_group("${_GENERATED_SOURCE_DIR}" "${_GENERATED_SOURCE_DIR}" "cpp" "_FILES_SOURCE")
list(APPEND _FILES_SOURCE "src/main.cpp")

# Always add all files to the target executable build call to add them to the build project.
alpaka_add_executable(
        ${_TARGET_NAME}
        ${_FILES_SOURCE})

target_link_libraries(${_TARGET_NAME} PUBLIC Catch2 Catch2WithMain)
target_compile_definitions(${_TARGET_NAME} PRIVATE "PMACC_NO_TPP_INCLUDE")
target_link_libraries(${_TARGET_NAME} PRIVATE pmacc::pmacc)

set_target_properties(headerCheckTest PROPERTIES FOLDER "headerCheck")

add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME})
