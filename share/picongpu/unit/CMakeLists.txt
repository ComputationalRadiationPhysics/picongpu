# Copyright 2023-2024 Rene Widera
#
# This file is part of PIConGPU.
#
# PIConGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PIConGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.
#

cmake_minimum_required(VERSION 3.25.0)
project("UnitTest")

###############################################################################
# Language Flags
###############################################################################

# enforce C++20
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)

################################################################################
# CMake policies
#
# Search in <PackageName>_ROOT:
#   https://cmake.org/cmake/help/v3.12/policy/CMP0074.html
################################################################################
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

################################################################################
# PMacc
################################################################################
find_package(PMacc REQUIRED CONFIG PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../../include/pmacc")

###############################################################################
# Catch2
###############################################################################

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../thirdParty/catch2 ${CMAKE_BINARY_DIR}/catch2)

################################################################################
# MPI
################################################################################

# MPI is provided by pmacc but to execute the binaries via root additional flags must be given to the execution command
option(USE_MPI_AS_ROOT_USER "add --allow-run-as-root mpiexec used by ctest" OFF)

if(USE_MPI_AS_ROOT_USER)
    set(MPI_RUNTIME_FLAGS "--allow-run-as-root")
endif()

###############################################################################
# Targets
###############################################################################

# CTest
enable_testing()

# Add subdirectories for tests
add_subdirectory(dimensional_tests)
add_subdirectory(non_dimensional_tests)
