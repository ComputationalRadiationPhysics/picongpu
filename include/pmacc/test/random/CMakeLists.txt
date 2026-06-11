# SPDX-FileCopyrightText: Alexander Grund
#
# SPDX-License-Identifier: LGPL-3.0-or-later

#

cmake_minimum_required(VERSION 3.28.0)
project("TestRandomGenerators")

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/../..")

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
find_package(PMacc REQUIRED CONFIG PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../..")

###############################################################################
# Targets
###############################################################################

alpaka_add_executable(TestRng 2DDistribution.cpp)
target_link_libraries(TestRng PRIVATE pmacc::pmacc)

add_custom_target(run COMMAND mpiexec -n 1 TestRng DEPENDS TestRng WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
