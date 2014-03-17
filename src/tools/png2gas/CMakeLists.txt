#
# Copyright 2014 Axel Huebl, Benjamin Schneider, Felix Schmitt, Heiko Burau, Rene Widera
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.
#

################################################################################
# Required cmake version
################################################################################

cmake_minimum_required(VERSION 2.8.5)


################################################################################
# Project 
################################################################################

project(png2gas)

# set helper pathes to find libraries and packages
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/" "$ENV{BOOST_ROOT}" "$ENV{MPI_ROOT}"
    "$ENV{HDF5_ROOT}")

# install prefix
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}" CACHE PATH "install prefix" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT) 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wno-pmf-conversions -Wno-deprecated")


################################################################################
# Build type (debug, release)
################################################################################

option(RELEASE "disable all debug asserts" OFF)
if(NOT RELEASE)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
    set(CMAKE_BUILD_TYPE Debug)
    add_definitions(-DDEBUG)
    message("building debug")
else()
    message("building release")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Werror")
endif(NOT RELEASE)


################################################################################
# Find Boost
################################################################################

find_package(Boost REQUIRED COMPONENTS program_options)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Boost_LIBRARIES})


################################################################################
# PNGWRITER
################################################################################

find_path(PIC_PNGWRITER_ROOT_DIR
    NAMES include/pngwriter.h
    PATHS ENV PNGWRITER_ROOT 
    DOC "pngwriter ROOT location")

if(PIC_PNGWRITER_ROOT_DIR)
    message(STATUS "Found PNGWriter: "${PIC_PNGWRITER_ROOT_DIR})

    include_directories(SYSTEM ${PIC_PNGWRITER_ROOT_DIR}/include)
    link_directories(${PIC_PNGWRITER_ROOT_DIR}/lib)
    set(LIBS ${LIBS} pngwriter png)

    # freetype support enabled?
    include(FindFreetype)
    if(FREETYPE_FOUND)
      include_directories(${FREETYPE_INCLUDE_DIRS})
      set(LIBS ${LIBS} ${FREETYPE_LIBRARIES})
    else(FREETYPE_FOUND)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNO_FREETYPE")
    endif(FREETYPE_FOUND)
else(PIC_PNGWRITER_ROOT_DIR)
    message(FATAL_ERROR "Could not find PNGWriter")
endif(PIC_PNGWRITER_ROOT_DIR)


################################################################################
# libSplash (+ hdf5 due to required headers)
################################################################################

find_path(PIC_SPLASH_ROOT_DIR
  NAMES include/splash/splash.h lib/libsplash.a
  PATHS ENV SPLASH_ROOT
  DOC "libSplash ROOT location (provides HDF5 output)"
)
set(PIC_SPLASH_VERSION_MIN "1.1.1")
set(PIC_SPLASH_VERSION_OK OFF)

if(PIC_SPLASH_ROOT_DIR)
    # find version
    file(STRINGS "${PIC_SPLASH_ROOT_DIR}/include/splash/version.hpp"
         PIC_SPLASH_VERSION_MAJOR_HPP REGEX "#define SPLASH_VERSION_MAJOR ")
    file(STRINGS "${PIC_SPLASH_ROOT_DIR}/include/splash/version.hpp"
         PIC_SPLASH_VERSION_MINOR_HPP REGEX "#define SPLASH_VERSION_MINOR ")
    file(STRINGS "${PIC_SPLASH_ROOT_DIR}/include/splash/version.hpp"
         PIC_SPLASH_VERSION_PATCH_HPP REGEX "#define SPLASH_VERSION_PATCH ")
    string(REGEX MATCH "([0-9]+)" PIC_SPLASH_VERSION_MAJOR
                                ${PIC_SPLASH_VERSION_MAJOR_HPP})
    string(REGEX MATCH "([0-9]+)" PIC_SPLASH_VERSION_MINOR
                                ${PIC_SPLASH_VERSION_MINOR_HPP})
    string(REGEX MATCH "([0-9]+)" PIC_SPLASH_VERSION_PATCH
                                ${PIC_SPLASH_VERSION_PATCH_HPP})

    set(PIC_SPLASH_VERSION "${PIC_SPLASH_VERSION_MAJOR}.${PIC_SPLASH_VERSION_MINOR}.${PIC_SPLASH_VERSION_PATCH}")

    # status output and check version
    message(STATUS "Found libSplash: ${PIC_SPLASH_ROOT_DIR} "
                   "(found version \"${PIC_SPLASH_VERSION}\")")
    if("${PIC_SPLASH_VERSION}" VERSION_LESS "${PIC_SPLASH_VERSION_MIN}")
        message(WARNING "libSplash version is smaller than the required version ${PIC_SPLASH_VERSION_MIN} - not using")
    else()
        set(PIC_SPLASH_VERSION_OK ON)
    endif()
endif(PIC_SPLASH_ROOT_DIR)

if(PIC_SPLASH_VERSION_OK)
    # require hdf5
    find_package(HDF5 REQUIRED)

    # add include information
    include_directories(SYSTEM ${HDF5_INCLUDE_DIRS})
    include_directories(SYSTEM ${PIC_SPLASH_ROOT_DIR}/include)

    # libSplash compiled with parallel support?
    file(STRINGS "${PIC_SPLASH_ROOT_DIR}/include/splash/splash.h" _splash_H_CONTENTS
         REGEX "#define SPLASH_SUPPORTED_PARALLEL ")
    string(REGEX MATCH "([0-9]+)" _splash_IS_PARALLEL "${_splash_H_CONTENTS}")

    # check that libSplash supports parallel and is compatible with hdf5
    if("${_splash_IS_PARALLEL}")
       message(STATUS "libSplash supports PARALLEL output")
       if(NOT HDF5_IS_PARALLEL)
           message(FATAL_ERROR "libSplash compiled with PARALLEL support but HDF5 lacks it...")
       endif(NOT HDF5_IS_PARALLEL)

        find_package(MPI REQUIRED)
        include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
        set(LIBS ${LIBS} ${MPI_C_LIBRARIES})

        # bullxmpi fails if it can not find its c++ counter part
        if(MPI_CXX_FOUND) 
            set(LIBS ${LIBS} ${MPI_CXX_LIBRARIES})
        endif(MPI_CXX_FOUND)
    else("${_splash_IS_PARALLEL}")
        message(FATAL_ERROR "Parallel libSplash is required")
    endif("${_splash_IS_PARALLEL}")

    # static
    add_library(splash_static STATIC IMPORTED)
    set_target_properties(splash_static PROPERTIES IMPORTED_LOCATION
                          ${PIC_SPLASH_ROOT_DIR}/lib/libsplash.a)
    set(LIBS ${LIBS} splash_static ${HDF5_LIBRARIES})
else(PIC_SPLASH_VERSION_OK)
    message(FATAL_ERROR "Could not find libSplash for hdf5 output")
endif(PIC_SPLASH_VERSION_OK)


################################################################################
# Compile & Link
################################################################################

file(GLOB SRCFILES "*.cpp")

add_executable(png2gas ${SRCFILES})

target_link_libraries (png2gas ${LIBS})


################################################################################
# Install
################################################################################

install(TARGETS png2gas RUNTIME DESTINATION .)
