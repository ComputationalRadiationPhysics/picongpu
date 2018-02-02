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

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Project.

SET(_TARGET_NAME "blockShared")

PROJECT(${_TARGET_NAME})

#-------------------------------------------------------------------------------
# Find alpaka test common.

SET(COMMON_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../common/" CACHE STRING  "The location of the alpaka test common library")
LIST(APPEND CMAKE_MODULE_PATH "${COMMON_ROOT}")
FIND_PACKAGE(common REQUIRED)

#-------------------------------------------------------------------------------
# Boost.Test.
FIND_PACKAGE(Boost QUIET COMPONENTS unit_test_framework)
IF(NOT Boost_UNIT_TEST_FRAMEWORK_FOUND)
    MESSAGE(FATAL_ERROR "Required alpaka dependency Boost.Test could not be found!")

ELSE()
    LIST(APPEND _INCLUDE_DIRECTORIES_PRIVATE ${Boost_INCLUDE_DIRS})
    LIST(APPEND _LINK_LIBRARIES_PRIVATE ${Boost_LIBRARIES})

ENDIF()

#-------------------------------------------------------------------------------
# Add executable.

SET(_SOURCE_DIR "src/")

# Add all the source files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_src_group("${_SOURCE_DIR}" "${_SOURCE_DIR}" "cpp" _FILES_SOURCE)

# Always add all files to the target executable build call to add them to the build project.
ALPAKA_ADD_EXECUTABLE(
    ${_TARGET_NAME}
    ${_FILES_SOURCE})
TARGET_COMPILE_OPTIONS(
    ${_TARGET_NAME}
    PRIVATE ${_DEFINITIONS_PRIVATE})
TARGET_INCLUDE_DIRECTORIES(
    ${_TARGET_NAME}
    PRIVATE ${_INCLUDE_DIRECTORIES_PRIVATE})
TARGET_LINK_LIBRARIES(
    ${_TARGET_NAME}
    PRIVATE common ${_LINK_LIBRARIES_PRIVATE})
IF(NOT Boost_USE_STATIC_LIBS)
    LIST(APPEND _DEFINITIONS_PRIVATE "BOOST_TEST_DYN_LINK")
        TARGET_COMPILE_DEFINITIONS(
            ${_TARGET_NAME}
            PRIVATE ${_DEFINITIONS_PRIVATE})
ENDIF()

# Group the targets into subfolders for IDEs supporting this.
SET_TARGET_PROPERTIES(${_TARGET_NAME} PROPERTIES FOLDER "test/unit")

ENABLE_TESTING()
ADD_TEST(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME})
