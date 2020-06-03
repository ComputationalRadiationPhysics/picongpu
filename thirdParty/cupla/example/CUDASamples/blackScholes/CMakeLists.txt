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

cmake_minimum_required(VERSION 3.11.4)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Project.
################################################################################

project("blackScholes")

################################################################################
# CMake policies
#
# Search in <PackageName>_ROOT:
#   https://cmake.org/cmake/help/v3.12/policy/CMP0074.html#
################################################################################

if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

################################################################################
# Find cupla
################################################################################

set(cupla_ROOT "$ENV{CUPLA_ROOT}" CACHE STRING  "The location of the cupla library")

list(APPEND CMAKE_MODULE_PATH "${cupla_ROOT}")
find_package("cupla" REQUIRED)


################################################################################
# Add library.
################################################################################

set(_SOURCE_DIR "src/")

# Add all the source files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_src_group("${_SOURCE_DIR}" "" "cpp" _FILES_SOURCE_CXX)

# Always add all files to the target executable build call to add them to the build project.
cupla_add_executable(${PROJECT_NAME} ${_FILES_SOURCE_CXX})

target_include_directories(${PROJECT_NAME} PRIVATE "../common/")
