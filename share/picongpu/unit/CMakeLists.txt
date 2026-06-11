# SPDX-FileCopyrightText: Rene Widera
#
# SPDX-License-Identifier: GPL-3.0-or-later

#

cmake_minimum_required(VERSION 3.28.0)
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
# Set the Cmake policy version. This sets the behaviour of the following policies
# Search in <PackageName>_ROOT:
#   https://cmake.org/cmake/help/latest/policy/CMP0074.html
# Search in <PACKAGENAME>_ROOT:
#   https://cmake.org/cmake/help/latest/policy/CMP0144.html
################################################################################
cmake_policy(VERSION 3.28.0)

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
