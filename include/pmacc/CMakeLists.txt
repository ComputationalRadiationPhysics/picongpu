# SPDX-FileCopyrightText: Erik Zenker, Alexander Grund
#
# SPDX-License-Identifier: LGPL-3.0-or-later

#
#

################################################################################
# PMacc tests
################################################################################

cmake_minimum_required(VERSION 3.28.0)
project(PMaccTest)

# set helper pathes to find libraries and packages
# Add specific hints
list(APPEND CMAKE_PREFIX_PATH "$ENV{MPI_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{CUDA_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{BOOST_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{VT_ROOT}")
# Add from environment after specific env vars
list(APPEND CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH}")

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

###############################################################################
# Language Flags
###############################################################################

# enforce C++20
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)

################################################################################
# Directory of this file.
################################################################################
set(PMACC_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Normalize the path (e.g. remove ../)
get_filename_component(PMACC_ROOT_DIR "${PMACC_ROOT_DIR}" ABSOLUTE)

################################################################################
# PMacc
################################################################################

find_package(PMacc REQUIRED CONFIG PATHS ${CMAKE_CURRENT_SOURCE_DIR})

###############################################################################
# Catch2
###############################################################################

add_subdirectory(${PMACC_ROOT_DIR}/../../thirdParty/catch2 ${CMAKE_BINARY_DIR}/catch2)

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

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/test)

# CTest
enable_testing()

# Test cases
# Each *UT.cpp file is an independent executable with one or more test cases
file(GLOB_RECURSE TESTS test/*UT.cpp)
foreach(dim 2 3)
    foreach(testCaseFilepath ${TESTS})
        get_filename_component(testCaseFilename ${testCaseFilepath} NAME)
        string(REPLACE "UT.cpp" "" testCase ${testCaseFilename})
        set(testExe "${PROJECT_NAME}-${testCase}-${dim}D")
        alpaka_add_executable(${testExe} ${testCaseFilepath})
        target_compile_definitions(${testExe} PRIVATE TEST_DIM=${dim})
        target_link_libraries(${testExe} PUBLIC Catch2 Catch2WithMain)
        target_link_libraries(${testExe} PRIVATE pmacc::pmacc)
        add_test(
            NAME "${testCase}-${dim}D"
            COMMAND mpiexec ${MPI_RUNTIME_FLAGS} -n 1 ./${testExe}
        )
    endforeach()
    string(REPLACE "-DTEST_DIM=${dim}" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endforeach()
