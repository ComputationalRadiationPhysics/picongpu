# - Find mallocMC library,
#     Memory Allocator for Many Core Architectures
#     https://github.com/ComputationalRadiationPhysics/mallocMC
#
# Use this module by invoking find_package with the form:
#   find_package(mallocMC
#     [version] [EXACT]     # Minimum or EXACT version, e.g. 2.0.0
#     [REQUIRED]            # Fail with an error if mallocMC or a required
#                           #   component is not found
#     [QUIET]               # Do not warn if this module was not found
#   )
#
# To provide a hint to this module where to find the mallocMC installation,
# set the MALLOCMC_ROOT environment variable. You can also set the
# MALLOCMC_ROOT CMake variable, which will take precedence over the environment
# variable.
#
# This module requires CUDA and Boost. When calling it, make sure to call
# find_package(CUDA) and find_package(Boost) first.
#
# This module will define the following variables:
#   mallocMC_INCLUDE_DIRS    - Include directories for the mallocMC headers.
#   mallocMC_FOUND           - TRUE if FindmallocMC found a working install
#   mallocMC_VERSION         - Version in format Major.Minor.Patch
#
# The following variables are optional and only defined if the selected
# components require them:
#   mallocMC_LIBRARIES       - mallocMC libraries for dynamic linking using
#                              target_link_libraries(${mallocMC_LIBRARIES})
#   mallocMC_DEFINITIONS     - Compiler definitions you should add with
#                              add_definitions(${mallocMC_DEFINITIONS})
#


###############################################################################
# Copyright 2014-2015 Axel Huebl, Felix Schmitt, Rene Widera,
#                     Carlchristian Eckert
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


# Required cmake version ######################################################
#
cmake_minimum_required(VERSION 2.8.5)


# mallocMC ####################################################################
#
set(mallocMC_FOUND TRUE)


# dependencies ################################################################
#
find_package(CUDA 5.0 REQUIRED)
find_package(Boost 1.48.0 REQUIRED)


# find mallocMC installation ##################################################
#
find_path(mallocMC_ROOT_DIR
    NAMES include/mallocMC/mallocMC.hpp
    PATHS ${MALLOCMC_ROOT} ENV MALLOCMC_ROOT
    PATH_SUFFIXES "src"
    DOC "mallocMC ROOT location"
    )

set(mallocMC_REQUIRED_VARS_LIST mallocMC_ROOT_DIR mallocMC_INCLUDE_DIRS)

if(mallocMC_ROOT_DIR)

    # find version ##############################################################
    #
    file(STRINGS "${mallocMC_ROOT_DIR}/include/mallocMC/version.hpp"
        mallocMC_VERSION_MAJOR_HPP REGEX "#define MALLOCMC_VERSION_MAJOR ")
    file(STRINGS "${mallocMC_ROOT_DIR}/include/mallocMC/version.hpp"
        mallocMC_VERSION_MINOR_HPP REGEX "#define MALLOCMC_VERSION_MINOR ")
    file(STRINGS "${mallocMC_ROOT_DIR}/include/mallocMC/version.hpp"
        mallocMC_VERSION_PATCH_HPP REGEX "#define MALLOCMC_VERSION_PATCH ")
    string(REGEX MATCH "([0-9]+)" mallocMC_VERSION_MAJOR
                                ${mallocMC_VERSION_MAJOR_HPP})
    string(REGEX MATCH "([0-9]+)" mallocMC_VERSION_MINOR
                                ${mallocMC_VERSION_MINOR_HPP})
    string(REGEX MATCH "([0-9]+)" mallocMC_VERSION_PATCH
                                ${mallocMC_VERSION_PATCH_HPP})

    # mallocMC variables ########################################################
    #
    set(mallocMC_VERSION "${mallocMC_VERSION_MAJOR}.${mallocMC_VERSION_MINOR}.${mallocMC_VERSION_PATCH}")
    set(mallocMC_INCLUDE_DIRS ${mallocMC_ROOT_DIR}/include)

    # check additional components ###############################################
    #
    foreach(COMPONENT ${mallocMC_FIND_COMPONENTS})
        set(mallocMC_${COMPONENT}_FOUND TRUE)

        if(${COMPONENT} STREQUAL "halloc")

            # halloc linked library #################################################
            #
            list(APPEND mallocMC_REQUIRED_VARS_LIST mallocMC_LIBRARIES)
            find_library(${COMPONENT}_LIBRARY
                NAMES libhalloc.a
                PATHS "${mallocMC_ROOT_DIR}/../halloc/" ENV HALLOC_ROOT
                PATH_SUFFIXES "lib" "bin"
                )
            if(${COMPONENT}_LIBRARY)
                list(APPEND mallocMC_LIBRARIES ${${COMPONENT}_LIBRARY})
            else(${COMPONENT}_LIBRARY)
                if(mallocMC_FIND_REQUIRED OR NOT mallocMC_FIND_QUIETLY)
                    message(WARNING "libhalloc.a not found. Ensure it is compiled correctly and set HALLOC_ROOT")
                endif()
                set(mallocMC_${COMPONENT}_FOUND FALSE)
            endif(${COMPONENT}_LIBRARY)

            # halloc headers ########################################################
            #
            find_path(${COMPONENT}_INCLUDE_DIR
                NAMES halloc.h
                PATHS "${mallocMC_ROOT_DIR}/../halloc/" ENV HALLOC_ROOT
                PATH_SUFFIXES "include" "src"
                )
            if(${COMPONENT}_INCLUDE_DIR)
                list(APPEND mallocMC_INCLUDE_DIRS ${${COMPONENT}_INCLUDE_DIR})
            else(${COMPONENT}_INCLUDE_DIR)
                set(mallocMC_${COMPONENT}_FOUND FALSE)
            endif(${COMPONENT}_INCLUDE_DIR)

            # set separable compilation #############################################
            #
            if(mallocMC_${COMPONENT}_FOUND)
                set(CUDA_SEPARABLE_COMPILATION ON PARENT_SCOPE)
            endif(mallocMC_${COMPONENT}_FOUND)

        endif(${COMPONENT} STREQUAL "halloc")

    endforeach(COMPONENT ${mallocMC_FIND_COMPONENTS})

endif(mallocMC_ROOT_DIR)


# handles the REQUIRED, QUIET and version-related arguments for find_package ##
#
list(REMOVE_DUPLICATES mallocMC_REQUIRED_VARS_LIST)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(mallocMC
    FOUND_VAR mallocMC_FOUND
    REQUIRED_VARS ${mallocMC_REQUIRED_VARS_LIST}
    VERSION_VAR mallocMC_VERSION
    HANDLE_COMPONENTS
    )
unset(mallocMC_REQUIRED_VARS_LIST)
