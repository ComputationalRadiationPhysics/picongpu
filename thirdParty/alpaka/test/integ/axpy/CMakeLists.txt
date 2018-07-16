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

SET(_TARGET_NAME "axpy")

PROJECT(${_TARGET_NAME})

#-------------------------------------------------------------------------------
# Find alpaka test common.

SET(COMMON_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../common/" CACHE STRING  "The location of the alpaka test common library")
LIST(APPEND CMAKE_MODULE_PATH "${COMMON_ROOT}")
FIND_PACKAGE(common REQUIRED)

#-------------------------------------------------------------------------------
# Add executable.

SET(_SOURCE_DIR "src/")

# Add all the source files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_src_group("${_SOURCE_DIR}" "${_SOURCE_DIR}" "cpp" _FILES_SOURCE)

# NVCC does not incorporate the COMPILE_OPTIONS of a target but only the CMAKE_CXX_FLAGS
IF(ALPAKA_ACC_GPU_CUDA_ENABLE AND NOT ALPAKA_CUDA_COMPILER MATCHES "clang")
    GET_TARGET_PROPERTY(_COMMON_COMPILE_OPTIONS common COMPILE_OPTIONS)
    STRING(REPLACE ";" " " _COMMON_COMPILE_OPTIONS_STRING "${_COMMON_COMPILE_OPTIONS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_COMMON_COMPILE_OPTIONS_STRING}")
ENDIF()

# Always add all files to the target executable build call to add them to the build project.
ALPAKA_ADD_EXECUTABLE(
    ${_TARGET_NAME}
    ${_FILES_SOURCE})
# Set the link libraries for this library (adds libs, include directories, defines and compile options).
TARGET_LINK_LIBRARIES(
    ${_TARGET_NAME}
    PUBLIC "common")

# Group the targets into subfolders for IDEs supporting this.
SET_TARGET_PROPERTIES(${_TARGET_NAME} PROPERTIES FOLDER "test/integ")
