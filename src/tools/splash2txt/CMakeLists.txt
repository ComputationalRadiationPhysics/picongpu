#
# Copyright 2013-2014 Felix Schmitt, Axel Huebl, Rene Widera
#
# This file is part of splash2txt. 
# 
# splash2txt is free software: you can redistribute it and/or modify 
# it under the terms of of either the GNU General Public License or 
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

cmake_minimum_required(VERSION 2.8.5)


################################################################################
# Project 
################################################################################

project(splash2txt)

set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/" "$ENV{BOOST_ROOT}" "$ENV{HDF5_ROOT}"
    "$ENV{MPI_ROOT}")

# own modules for find_packages
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/)

include_directories(include)


################################################################################
# Build type (debug, release)
################################################################################

option(S2T_RELEASE "disable all debug asserts" ON)
if(NOT S2T_RELEASE)
    set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall")
    set(CMAKE_BUILD_TYPE Debug)
endif(NOT S2T_RELEASE)


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
# prefer static libraries over shared ones (but do not force them)
set(Splash_USE_STATIC_LIBS ON)
find_package(Splash 1.2.0 REQUIRED COMPONENTS PARALLEL)

if(Splash_FOUND)
    include_directories(SYSTEM ${Splash_INCLUDE_DIRS})
    list(APPEND Splash_DEFINITIONS "-DENABLE_HDF5=1")
    add_definitions(${Splash_DEFINITIONS})
    message(STATUS "libSplash compiler definitions: ${Splash_DEFINITIONS}")
    set(LIBS ${LIBS} ${Splash_LIBRARIES})
endif(Splash_FOUND)


################################################################################
# Find Boost
################################################################################

find_package(Boost REQUIRED COMPONENTS program_options regex)

include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIR})
set(LIBS ${LIBS} ${Boost_LIBRARIES})


################################################################################
# Compile & Link splash2txt
################################################################################

file(GLOB SRCFILES "*.cpp")

add_executable(splash2txt
     ${SRCFILES}
     )

target_link_libraries(splash2txt m ${LIBS})


################################################################################
# Install
################################################################################

install(TARGETS splash2txt RUNTIME DESTINATION .)
