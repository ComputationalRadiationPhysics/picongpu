# Copyright 2013-2021 Felix Schmitt, Axel Huebl, Rene Widera
#
# This file is part of splash2txt.
#
# splash2txt is free software: you can redistribute it and/or modify
# it under the terms of either the GNU General Public License or
# the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# splash2txt is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License and the GNU Lesser General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# and the GNU Lesser General Public License along with splash2txt.
# If not, see <http://www.gnu.org/licenses/>.
#

################################################################################
# Required cmake version
################################################################################

cmake_minimum_required(VERSION 3.15.0)


################################################################################
# Project
################################################################################

project(splash2txt)

# Add specific hints
list(APPEND CMAKE_PREFIX_PATH "$ENV{MPI_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{BOOST_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{HDF5_ROOT}")
# Add from environment after specific env vars
list(APPEND CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH}")
# Last add generic system path to the end (as last fallback)
list(APPEND "/usr/lib/x86_64-linux-gnu/")

# own modules for find_packages
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../thirdParty/cmake-modules/)

include_directories(include)

################################################################################
# CMake policies
#
# Search in <PackageName>_ROOT:
#   https://cmake.org/cmake/help/v3.12/policy/CMP0074.html
################################################################################

if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()


###############################################################################
# Language Flags
###############################################################################

# enforce C++14
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 14)


################################################################################
# Build type (debug, release, relwithdebinfo)
################################################################################

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Debug' as none was specified.")
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wshadow")

if(CMAKE_BUILD_TYPE MATCHES Debug)
elseif(CMAKE_BUILD_TYPE MATCHES Release)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Werror")
elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -Werror")
else()
  message(FATAL_ERROR "Unrecognized build type")
endif()

message(STATUS "Building ${CMAKE_BUILD_TYPE}")

################################################################################
# Find MPI
################################################################################

find_package(MPI)
include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
set(LIBS ${LIBS} ${MPI_C_LIBRARIES})

# bullxmpi fails if it can not find its c++ counter part
if(MPI_CXX_FOUND)
    set(LIBS ${LIBS} ${MPI_CXX_LIBRARIES})
endif(MPI_CXX_FOUND)


################################################################################
# libSplash (+ hdf5 due to required headers)
################################################################################

# find libSplash installation
find_package(Splash 1.7.0 REQUIRED CONFIG COMPONENTS PARALLEL)

add_definitions(-DENABLE_HDF5=1)
message(STATUS "Found Splash: ${Splash_DIR}")
set(LIBS ${LIBS} Splash::Splash)


################################################################################
# Find zlib
################################################################################

# not directly needed, yet weirdly injected without target by Splash 1.7.0
find_package(ZLIB REQUIRED)


################################################################################
# ADIOS
################################################################################

# find adios installation
#   set(ADIOS_USE_STATIC_LIBS ON) # force static linking
find_package(ADIOS 1.13.1)

if(ADIOS_FOUND)
    add_definitions(-DENABLE_ADIOS=1)

    include_directories(SYSTEM ${ADIOS_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${ADIOS_LIBRARIES})
endif(ADIOS_FOUND)


################################################################################
# Find Boost
################################################################################

find_package(Boost 1.65.1 REQUIRED COMPONENTS program_options)
if(TARGET Boost::program_options)
    set(LIBS ${LIBS} Boost::boost Boost::program_options)
else()
    include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${Boost_LIBRARIES})
endif()


################################################################################
# Find math from stdlib
################################################################################

if(NOT WIN32 AND NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # automatically added on windows
    # should not be added for Intel compiler as it has an optimized lib included
    set(LIBS ${LIBS} m)
endif()


################################################################################
# Compile & Link splash2txt
################################################################################

set(SRCFILES "splash2txt.cpp")

list(APPEND SRCFILES "tools_splash_parallel.cpp")
if(ADIOS_FOUND)
    list(APPEND SRCFILES "tools_adios_parallel.cpp")
endif(ADIOS_FOUND)

add_executable(splash2txt
     ${SRCFILES}
     )

target_link_libraries(splash2txt PRIVATE ${LIBS})


################################################################################
# Install
################################################################################

install(TARGETS splash2txt RUNTIME DESTINATION .)
