# - Find libSplash library,
#     Simple Parallel file output Library for Accumulating Simulation data
#     using Hdf5 
#     https://github.com/ComputationalRadiationPhysics/libSplash
#
# Use this module by invoking find_package with the form:
#   find_package(Splash
#     [version] [EXACT]     # Minimum or EXACT version, e.g. 1.1.1
#     [REQUIRED]            # Fail with an error if Splash or a required
#                           #   component is not found
#     [QUIET]               # Do not warn if this module was not found
#     [COMPONENTS <...>]    # Compiled in components: PARALLEL
#   )
#
# To provide a hint to this module where to find the Splash installation,
# set the SPLASH_ROOT environment variable.
#
# This module requires HDF5. Make sure to provide a valid install of it
# under the environment variable HDF5_ROOT.
# Parallel HDF5/libSplash will require MPI (set the environment MPI_ROOT).
#
# Set the following CMake variables BEFORE calling find_packages to
# change the behavior of this module:
#   Splash_USE_STATIC_LIBS - Set to ON to force linking to the static
#                            library and its dependencies. Default: OFF
#
# This module will define the following variables:
#   Splash_INCLUDE_DIRS    - Include directories for the Splash headers.
#   Splash_LIBRARIES       - Splash libraries.
#   Splash_FOUND           - TRUE if FindSplash found a working install
#   Splash_VERSION         - Version in format Major.Minor.Patch
#   Splash_IS_PARALLEL     - Does this install support parallel IO?
#   Splash_DEFINITIONS     - Compiler definitions you should add with
#                            add_definitions(${Splash_DEFINITIONS})
#


###############################################################################
# Copyright 2014 Axel Huebl, Felix Schmitt, Rene Widera
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
###############################################################################


###############################################################################
# Required cmake version
###############################################################################

cmake_minimum_required(VERSION 2.8.5)


###############################################################################
# libSplash
###############################################################################

# we start by assuming we found Splash and falsify it if some
# dependencies are missing (or if we did not find Splash at all)
set(Splash_FOUND TRUE)

# option: use only static libs ################################################
#
if(Splash_USE_STATIC_LIBS)
    # carfully: we have to restore the original path in the end
    set(_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
    set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
    # link HDF5 static, too
    set(HDF5_USE_STATIC_LIBRARIES ON)
endif()

# find libSplash installation #################################################
#
find_path(Splash_ROOT_DIR
  NAMES include/splash/splash.h lib/libsplash.a
  PATHS ENV SPLASH_ROOT
  DOC "libSplash ROOT location (provides HDF5 output)"
)

if(Splash_ROOT_DIR)
    # Splash headers ##########################################################
    #
    list(APPEND Splash_INCLUDE_DIRS ${Splash_ROOT_DIR}/include)

    # Splash definitions ######################################################
    #

    # Splash libraries ########################################################
    #
    find_library(Splash_LIBRARIES
      NAMES splash
      PATHS $ENV{SPLASH_ROOT}/lib)

    # find version ############################################################
    #
    file(STRINGS "${Splash_ROOT_DIR}/include/splash/version.hpp"
         Splash_VERSION_MAJOR_HPP REGEX "#define SPLASH_VERSION_MAJOR ")
    file(STRINGS "${Splash_ROOT_DIR}/include/splash/version.hpp"
         Splash_VERSION_MINOR_HPP REGEX "#define SPLASH_VERSION_MINOR ")
    file(STRINGS "${Splash_ROOT_DIR}/include/splash/version.hpp"
         Splash_VERSION_PATCH_HPP REGEX "#define SPLASH_VERSION_PATCH ")
    string(REGEX MATCH "([0-9]+)" Splash_VERSION_MAJOR
                                ${Splash_VERSION_MAJOR_HPP})
    string(REGEX MATCH "([0-9]+)" Splash_VERSION_MINOR
                                ${Splash_VERSION_MINOR_HPP})
    string(REGEX MATCH "([0-9]+)" Splash_VERSION_PATCH
                                ${Splash_VERSION_PATCH_HPP})

    set(Splash_VERSION "${Splash_VERSION_MAJOR}.${Splash_VERSION_MINOR}.${Splash_VERSION_PATCH}")

    # require hdf5 ############################################################
    #
    find_package(HDF5 REQUIRED)
    list(APPEND Splash_INCLUDE_DIRS ${HDF5_INCLUDE_DIRS})
    list(APPEND Splash_DEFINITIONS ${HDF5_DEFINITIONS})
    list(APPEND Splash_LIBRARIES ${HDF5_LIBRARIES})

    # libSplash compiled with parallel support? ###############################
    #
    file(STRINGS "${Splash_ROOT_DIR}/include/splash/splash.h" _splash_H_CONTENTS
         REGEX "#define SPLASH_SUPPORTED_PARALLEL ")
    string(REGEX MATCH "([0-9]+)" Splash_IS_PARALLEL "${_splash_H_CONTENTS}")

    # check that libSplash supports parallel and is compatible with hdf5
    # if necesary find MPI, too
    if("${Splash_IS_PARALLEL}")
       set(Splash_IS_PARALLEL TRUE)
       message(STATUS "libSplash supports PARALLEL output")

       if(NOT HDF5_IS_PARALLEL)
           set(Splash_IS_PARALLEL FALSE)
           message(STATUS "libSplash compiled with PARALLEL support but HDF5 lacks it...")

       else(NOT HDF5_IS_PARALLEL)
           find_package(MPI REQUIRED)
           list(APPEND Splash_INCLUDE_DIRS ${MPI_C_INCLUDE_PATH})
           list(APPEND Splash_LIBRARIES ${MPI_C_LIBRARIES})
           # bullxmpi fails if it can not find its c++ counter part
           if(MPI_CXX_FOUND)
               list(APPEND Splash_LIBRARIES ${MPI_CXX_LIBRARIES})
           endif(MPI_CXX_FOUND)
       endif(NOT HDF5_IS_PARALLEL)

    endif("${Splash_IS_PARALLEL}")

    # COMPONENTS list #########################################################
    #
    set(Splash_MISSING_COMPONENTS "")
    #message(STATUS "SPLASH required components: ${Splash_FIND_COMPONENTS}")
    foreach(COMPONENT ${Splash_FIND_COMPONENTS})
        string(TOUPPER ${COMPONENT} COMPONENT)
        list(APPEND Splash_MISSING_COMPONENTS ${COMPONENT})
    endforeach()

    # remove components we FOUND ##############################################
    #
    if(Splash_IS_PARALLEL)
        list(REMOVE_ITEM Splash_MISSING_COMPONENTS "PARALLEL")
    endif(Splash_IS_PARALLEL)

    # all COMPONENTS found ? ##################################################
    #
    list(LENGTH Splash_MISSING_COMPONENTS Splash_NUM_COMPONENTS_MISSING)
    if(NOT ${Splash_NUM_COMPONENTS_MISSING} EQUAL 0)
        set(Splash_FOUND FALSE)
        message(STATUS "Missing COMPONENTS in libSplash: ${Splash_MISSING_COMPONENTS}")
    endif()

else(Splash_ROOT_DIR)
    set(Splash_FOUND FALSE)
    message(STATUS "Can NOT find libSplash for HDF5 output - set SPLASH_ROOT")
endif(Splash_ROOT_DIR)


# unset checked variables if not found ########################################
#
if(NOT Splash_FOUND)
    unset(Splash_INCLUDE_DIRS)
    unset(Splash_LIBRARIES)
    unset(Splash_IS_PARALLEL)
endif(NOT Splash_FOUND)


# restore CMAKE_FIND_LIBRARY_SUFFIXES if manipulated by this module ###########
#
if(Splash_USE_STATIC_LIBS)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
    unset(HDF5_USE_STATIC_LIBRARIES)
endif()


###############################################################################
# FindPackage Options
###############################################################################

# handles the REQUIRED, QUIET and version-related arguments for find_package
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Splash
    REQUIRED_VARS Splash_LIBRARIES Splash_INCLUDE_DIRS
    VERSION_VAR Splash_VERSION
)
