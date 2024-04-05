#
# Copyright 2013-2023 Axel Huebl, Rene Widera, Felix Schmitt
#
# This file is part of mpiInfo.
#
# mpiInfo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# mpiInfo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with mpiInfo.
# If not, see <http://www.gnu.org/licenses/>.
#

################################################################################
# Required cmake version
################################################################################

cmake_minimum_required(VERSION 3.22.0)


################################################################################
# Project
################################################################################

project(mpiInfo LANGUAGES CXX)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "install prefix" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# set helper pathes to find libraries and packages
# Add specific hints
list(APPEND CMAKE_PREFIX_PATH "$ENV{MPI_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{BOOST_ROOT}")
# Add from environment after specific env vars
list(APPEND CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH}")
# Last add generic system path to the end (as last fallback)
list(APPEND "/usr/lib/x86_64-linux-gnu/")


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
# mpiInfo options
################################################################################

option(MPIINFO_ADD_RPATH "Add RPATH's to binary." ON)


###############################################################################
# Language Flags
###############################################################################

# enforce C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)


################################################################################
# Warnings
################################################################################

set(CMAKE_CXX_FLAGS_DEFAULT "-Wall")

################################################################################
# Find Threads
################################################################################

# Allow users to override the "-pthread" preference.
if(NOT THREADS_PREFER_PTHREAD_FLAG)
     set(THREADS_PREFER_PTHREAD_FLAG TRUE)
endif()

# WARNING: in case were we cross compile we need to add '-DDMPI_CXX_WORKS=ON` and
# provide include flags and linker flags manually.
find_package(Threads REQUIRED)
if(NOT APPLE)
    # librt: undefined reference to `clock_gettime'
    find_library(RT_LIBRARY rt)
endif()


################################################################################
# Find MPI
################################################################################

find_package(MPI REQUIRED)

################################################################################
# Find Boost
################################################################################

find_package(Boost 1.66.0 REQUIRED COMPONENTS program_options)
if(TARGET Boost::program_options)
    set(MPIINFO_LIBS ${MPIINFO_LIBS} Boost::boost Boost::program_options)
else()
    include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
    set(MPIINFO_LIBS ${MPIINFO_LIBS} ${Boost_LIBRARIES})
endif()


################################################################################
# Compile & Link mpiInfo
################################################################################

add_executable(mpiInfo
    mpiInfo.cpp
)

target_link_libraries(mpiInfo PRIVATE ${MPIINFO_LIBS} Threads::Threads)
if(RT_LIBRARY AND NOT APPLE)
    # MPI runtime libraries
    target_link_libraries(mpiInfo PRIVATE ${RT_LIBRARY})
endif()
target_link_libraries(mpiInfo PRIVATE MPI::MPI_CXX)

if( CMAKE_TRY_COMPILE_TARGET_TYPE STREQUAL "STATIC_LIBRARY" AND CMAKE_EXE_LINKER_FLAGS)
    # Workaround for linker issues when linking static MPI libraries.
    # Because of CMAKE_TRY_COMPILE_TARGET_TYPE CMake providing the statics libraries before the object file from
    # `add_executable` therefore MPI, symbols can not be resolved. Linking the linker flags to the target again will
    # workaround the issue.
    target_link_libraries(mpiInfo PRIVATE ${CMAKE_EXE_LINKER_FLAGS})
endif()

## annotate with RPATH's
if(MPIINFO_ADD_RPATH)
    if(NOT DEFINED CMAKE_INSTALL_RPATH)
        if(APPLE)
            set_target_properties(mpiInfo PROPERTIES INSTALL_RPATH "@loader_path")
        elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
            set_target_properties(mpiInfo PROPERTIES INSTALL_RPATH "$ORIGIN")
        endif()
    endif()
    if(NOT DEFINED CMAKE_INSTALL_RPATH_USE_LINK_PATH)
        set_target_properties(mpiInfo PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
    endif()
endif()


################################################################################
# Install mpiInfo
################################################################################

install(TARGETS mpiInfo
         RUNTIME DESTINATION bin)
