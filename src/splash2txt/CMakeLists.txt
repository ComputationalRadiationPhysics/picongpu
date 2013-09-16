#
# Copyright 2013 Felix Schmitt, Axel Huebl, René Widera
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

cmake_minimum_required(VERSION 2.8.5)

project (splash2txt)
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/" "$ENV{BOOST_ROOT}" "$ENV{HDF5_ROOT}")

OPTION(S2T_RELEASE "disable all debug asserts" ON)
IF(NOT S2T_RELEASE)
    SET(CMAKE_CXX_FLAGS_DEBUG "-g -Wall")
    SET(CMAKE_BUILD_TYPE Debug)
ENDIF(NOT S2T_RELEASE)

FIND_PACKAGE(HDF5 REQUIRED)
INCLUDE_DIRECTORIES(${HDF5_INCLUDE_DIR})
LINK_DIRECTORIES(${HDF5_LIBRARY_DIRS})
SET(LIBS ${LIBS} hdf5)

#LIBSPLASH
find_path(S2T_LIBSPLASH_ROOT_DIR
  NAMES include/sdc_defines.hpp lib/libsplash.a
  PATHS ENV SPLASH_ROOT
  DOC "libsplash ROOT location. (only used if HDF5 is enabled)"
  )
INCLUDE_DIRECTORIES("${S2T_LIBSPLASH_ROOT_DIR}/include")
ADD_LIBRARY(splash_static STATIC IMPORTED)
SET_TARGET_PROPERTIES(splash_static PROPERTIES
	IMPORTED_LOCATION "${S2T_LIBSPLASH_ROOT_DIR}/lib/libsplash.a")
SET(LIBS ${LIBS} splash_static)

#Boost from system
FIND_PACKAGE(Boost REQUIRED COMPONENTS program_options regex)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})
SET(LIBS ${LIBS} ${Boost_LIBRARIES})


FILE(GLOB SRCFILES "*.cpp")

ADD_EXECUTABLE(splash2txt
     ${SRCFILES}
     )

TARGET_LINK_LIBRARIES(splash2txt m ${LIBS})

