#
# Copyright 2016 Rene Widera, Benjamin Worpitz, Vincent Ridder
#
# This file is part of cupla.
#
# cupla is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cupla is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with cupla.
# If not, see <http://www.gnu.org/licenses/>.
#


################################################################################
# Required CMake version.
################################################################################

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Project.
################################################################################

SET(_INCLUDE_DIR "../common/")
SET(_SUFFIXED_INCLUDE_DIR "../common/")
SET(_SOURCE_DIR "src/")

PROJECT("vecorAdd")

################################################################################
# Find cupla
################################################################################

SET(cupla_ROOT "$ENV{CUPLA_ROOT}" CACHE STRING  "The location of the cupla library")

LIST(APPEND CMAKE_MODULE_PATH "${cupla_ROOT}")
FIND_PACKAGE("cupla" REQUIRED)


################################################################################
# Add library.
################################################################################

# Add all the include files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_src_group("${_SUFFIXED_INCLUDE_DIR}" "" "hpp" _FILES_HEADER)
append_recursive_files_add_to_src_group("${_SUFFIXED_INCLUDE_DIR}" "" "h" _FILES_HEADER)

# Add all the source files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_src_group("${_SOURCE_DIR}" "" "cpp" _FILES_SOURCE_CXX)

set(_INCLUDE_DIRECTORIES_PRIVATE ${_INCLUDE_DIR})

include_directories(${_INCLUDE_DIRECTORIES_PRIVATE})

# Always add all files to the target executable build call to add them to the build project.
cupla_add_executable(
    "vectorAdd"
    ${_FILES_SOURCE_CXX})

# Set the link libraries for this library (adds libs, include directories, defines and compile options).
target_link_libraries(
    "vectorAdd"
    PUBLIC ${cupla_LIBRARIES})
