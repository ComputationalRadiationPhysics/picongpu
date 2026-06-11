# SPDX-FileCopyrightText: Rene Widera, Axel Huebl, Tapish Narwal
#
# SPDX-License-Identifier: LGPL-3.0-or-later

#

################################################################################
# Required cmake version
################################################################################

cmake_minimum_required(VERSION 3.28)

################################################################################
# Project
################################################################################

project(heatEq)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "install prefix" FORCE)
endif()

# set helper pathes to find libraries and packages
# Add specific hints
list(APPEND CMAKE_PREFIX_PATH "$ENV{MPI_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{BOOST_ROOT}")

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
# Find PMacc
################################################################################

find_package(PMacc REQUIRED CONFIG PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/pmacc")

# ################################################################################
# # Build type (debug, release)
# ################################################################################

option(HEATEQ_RELEASE "Build release version, disables all runtime asserts" OFF)
if(HEATEQ_RELEASE)
    message(STATUS "Release version")
    set(CMAKE_BUILD_TYPE Release)
    add_definitions(-DNDEBUG)
else(HEATEQ_RELEASE)
    message(STATUS "Debug version")
    set(CMAKE_BUILD_TYPE Debug)
endif(HEATEQ_RELEASE)

################################################################################
# PNGwriter
################################################################################

# find PNGwriter installation
find_package(PNGwriter 0.7.0 REQUIRED CONFIG)

if(PNGwriter_FOUND)
    list(APPEND PNGwriter_DEFINITIONS "-DGOL_ENABLE_PNG=1")
endif(PNGwriter_FOUND)

################################################################################
# Warnings
################################################################################

# GNU
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
    # new warning in gcc 4.8 (flag ignored in previous version)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-local-typedefs")
    # ICC
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    # PGI
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Minform=inform")
endif()

################################################################################
# Compile & Link  heatEq
################################################################################

include_directories(include)

file(GLOB SRCFILES "*.cpp")

alpaka_add_executable(heatEq
    ${SRCFILES}
)

target_link_libraries(heatEq PRIVATE pmacc::pmacc)
if(PNGwriter_FOUND)
    target_link_libraries(heatEq PRIVATE PNGwriter::PNGwriter)
endif()

################################################################################
# Math
################################################################################

if(NOT WIN32)
    # automatically added on windows
    target_link_libraries(heatEq PRIVATE m)
endif()

################################################################################
# Install heatEq
################################################################################

install(TARGETS heatEq RUNTIME DESTINATION bin)
