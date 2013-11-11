#
# Copyright 2013 Felix Schmitt, Axel Huebl, Rene Widera
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

set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/" "$ENV{MPI_ROOT}"
    "$ENV{BOOST_ROOT}" "$ENV{HDF5_ROOT}")


################################################################################
# Build type (debug, release)
################################################################################

option(S2T_RELEASE "disable all debug asserts" ON)
if(NOT S2T_RELEASE)
    set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall")
    set(CMAKE_BUILD_TYPE Debug)
endif(NOT S2T_RELEASE)


################################################################################
# Find HDF5
################################################################################

find_package(HDF5 REQUIRED)

include_directories(${HDF5_INCLUDE_DIR})
link_directories(${HDF5_LIBRARY_DIRS})
set(LIBS ${LIBS} hdf5)


################################################################################
# Find MPI (optional, required by parallel hdf5)
################################################################################

find_package(MPI)
if(MPI_C_FOUND)
    include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
    set(LIBS ${LIBS} ${MPI_C_LIBRARIES})
endif(MPI_C_FOUND)

# bullxmpi fails if it can not find its c++ counter part
if(MPI_CXX_FOUND)
    set(LIBS ${LIBS} ${MPI_CXX_LIBRARIES})
endif(MPI_CXX_FOUND)


################################################################################
# Find libSplash
################################################################################

find_path(S2T_LIBSPLASH_ROOT_DIR
  NAMES include/sdc_defines.hpp lib/libsplash.a
  PATHS ENV SPLASH_ROOT
  DOC "libSplash ROOT location. (only used if HDF5 is enabled)"
  )

include_directories("${S2T_LIBSPLASH_ROOT_DIR}/include")
add_library(splash_static STATIC IMPORTED)
set_target_properties(splash_static PROPERTIES
	IMPORTED_LOCATION "${S2T_LIBSPLASH_ROOT_DIR}/lib/libsplash.a")
set(LIBS ${LIBS} splash_static)


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
