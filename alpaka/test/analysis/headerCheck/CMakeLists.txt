#
# Copyright 2014-2019 Benjamin Worpitz, Axel Huebl
#
# This file is part of Alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

IF(NOT ALPAKA_CI OR (ALPAKA_CI AND ALPAKA_CI_ANALYSIS))

SET(_TARGET_NAME "headerCheck")

#-------------------------------------------------------------------------------
# Create source files.

SET(_ALPAKA_INCLUDE_DIRECTORY "${_ALPAKA_ROOT_DIR}/include")
SET(_ALPAKA_SUFFIXED_INCLUDE_DIR "${_ALPAKA_INCLUDE_DIRECTORY}/alpaka")
append_recursive_files("${_ALPAKA_SUFFIXED_INCLUDE_DIR}" "hpp" "_ALPAKA_FILES_HEADER")

SET(_GENERATED_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/src")

FILE(REMOVE_RECURSE ${_GENERATED_SOURCE_DIR})

FOREACH(_HEADER_FILE ${_ALPAKA_FILES_HEADER})
    # Remove the parent directory from the path.
    # NOTE: This is not correct because it does not only replace at the beginning of the string.
    #  "STRING(REGEX REPLACE" would be correct if there was an easy way to escape arbitrary strings.
    STRING(
        REPLACE "${_ALPAKA_SUFFIXED_INCLUDE_DIR}/" ""
        _HEADER_FILE
        "${_HEADER_FILE}")
    SET(_SOURCE_FILE "${_GENERATED_SOURCE_DIR}/${_HEADER_FILE}.cpp")
    FILE(WRITE "${_SOURCE_FILE}" "#include <alpaka/${_HEADER_FILE}>\n#include <alpaka/${_HEADER_FILE}>\n")
ENDFOREACH()

#-------------------------------------------------------------------------------
# Add executable.

append_recursive_files_add_to_src_group("${_GENERATED_SOURCE_DIR}" "${_GENERATED_SOURCE_DIR}" "cpp" "_FILES_SOURCE")
LIST(APPEND _FILES_SOURCE "src/main.cpp")

# Always add all files to the target executable build call to add them to the build project.
ALPAKA_ADD_EXECUTABLE(
    ${_TARGET_NAME}
    ${_FILES_SOURCE})
TARGET_INCLUDE_DIRECTORIES(
    ${_TARGET_NAME}
    PRIVATE ${Boost_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(
    ${_TARGET_NAME}
    PRIVATE common)

SET_TARGET_PROPERTIES(headerCheck PROPERTIES FOLDER "test/analysis")

ADD_TEST(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} ${_ALPAKA_TEST_OPTIONS})

ENDIF()
