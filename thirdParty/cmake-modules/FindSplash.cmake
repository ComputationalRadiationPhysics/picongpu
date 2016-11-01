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
# set CMAKE_PREFIX_PATH or the SPLASH_ROOT environment variable.
#
# This module requires HDF5. Make sure to provide a valid install of it
# via CMAKE_PREFIX_PATH or under the environment variable HDF5_ROOT.
# Parallel HDF5/libSplash will require MPI (set the environment MPI_ROOT).
#
# Set the following CMake variables BEFORE calling find_packages to
# change the behavior of this module:
#   Splash_USE_STATIC_LIBS - Set to ON to prefer linking to the static
#                            library and its static dependencies. Note that it
#                            can fall back to shared libraries. Default: OFF
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
# Copyright 2014-2016 Axel Huebl, Felix Schmitt, Rene Widera, Alexander Grund
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
# RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE
# USE OR PERFORMANCE OF THIS SOFTWARE.
###############################################################################


###############################################################################
# Required cmake version
###############################################################################

cmake_minimum_required(VERSION 2.8.11)


###############################################################################
# libSplash
###############################################################################

# we start by assuming we found Splash and falsify it if some
# dependencies are missing (or if we did not find Splash at all)
set(Splash_FOUND TRUE)

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

    # option: prefer static libs ##############################################
    #
    if(Splash_USE_STATIC_LIBS)
        # carefully: we have to restore the original path in the end
        set(_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
        set(CMAKE_FIND_LIBRARY_SUFFIXES .a .so)
    endif()

    # Splash libraries ########################################################
    #
    find_library(Splash_LIBRARIES
      NAMES splash
      PATHS ${Splash_ROOT_DIR}/lib)

    # restore CMAKE_FIND_LIBRARY_SUFFIXES if manipulated by this module #######
    #
    if(Splash_USE_STATIC_LIBS)
        set(CMAKE_FIND_LIBRARY_SUFFIXES ${_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
    endif()

    # require hdf5 ############################################################
    #
    # disabled: Some libraries like libdl are also included by MPI and would
    #           be linked once static once dynamic. Upgrading MPI to static
    #           binding is too intrusive so we stay with only libSplash static
    #           and everything else "default".
    #if(Splash_USE_STATIC_LIBS)
    #    set(HDF5_USE_STATIC_LIBRARIES ON)
    #endif()
    find_package(HDF5 REQUIRED COMPONENTS C)
    list(APPEND Splash_INCLUDE_DIRS ${HDF5_INCLUDE_DIRS})
    list(APPEND Splash_DEFINITIONS ${HDF5_DEFINITIONS})
    list(APPEND Splash_LIBRARIES ${HDF5_LIBRARIES})

    # restore old settings
    #if(Splash_USE_STATIC_LIBS)
    #    unset(HDF5_USE_STATIC_LIBRARIES)
    #endif()

    # libSplash compiled with parallel support? ###############################
    #
    file(STRINGS "${Splash_ROOT_DIR}/include/splash/splash.h" _splash_H_CONTENTS
         REGEX "#define SPLASH_SUPPORTED_PARALLEL ")
    string(REGEX MATCH "([0-9]+)" Splash_IS_PARALLEL "${_splash_H_CONTENTS}")

    # check that libSplash supports parallel and is compatible with hdf5
    # if necessary find MPI, too
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

else(Splash_ROOT_DIR)
    set(Splash_FOUND FALSE)
    message(STATUS "Can NOT find libSplash for HDF5 output - include its root in CMAKE_PREFIX_PATH")
endif(Splash_ROOT_DIR)


# unset checked variables if not found ########################################
#
if(NOT Splash_FOUND)
    unset(Splash_INCLUDE_DIRS)
    unset(Splash_LIBRARIES)
    unset(Splash_IS_PARALLEL)
endif(NOT Splash_FOUND)


###############################################################################
# FindPackage Options
###############################################################################

# handles the REQUIRED, QUIET and version-related arguments for find_package
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Splash
    FOUND_VAR Splash_FOUND
    REQUIRED_VARS Splash_LIBRARIES Splash_INCLUDE_DIRS
    VERSION_VAR Splash_VERSION
)
