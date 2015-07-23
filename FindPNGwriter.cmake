# - Find PNGwriter library,
#     a C++ library for creating PNG images
#     https://github.com/ax3l/pngwriter
#
# Use this module by invoking find_package with the form:
#   find_package(PNGwriter
#     [version] [EXACT]     # Minimum or EXACT version, e.g. 0.5.4
#     [REQUIRED]            # Fail with an error if PNGwriter or a required
#                           #   component is not found
#     [QUIET]               # Do not warn if this module was not found
#     [COMPONENTS <...>]    # Compiled in components: not implemented
#   )
#
# To provide a hint to this module where to find the PNGwriter installation,
# set the PNGWRITER_ROOT environment variable.
#
# This module requires a valid installation of libpng (1.2.9+) which in turn
# requires ZLib.
#
# Set the following CMake variables BEFORE calling find_packages to
# change the behavior of this module:
#   PNGwriter_USE_STATIC_LIBS - Set to ON to prefer linking to the static
#                               library and its static dependencies. Note that it
#                               can fall back to shared libraries. Default: OFF
#
# This module will define the following variables:
#   PNGwriter_INCLUDE_DIRS    - Include directories for the PNGwriter headers.
#   PNGwriter_LIBRARIES       - PNGwriter libraries.
#   PNGwriter_FOUND           - TRUE if FindPNGwriter found a working install
#   PNGwriter_VERSION         - Version in format Major.Minor.Patch
#   PNGwriter_HAS_FREETYPE    - Does this install require a freetype install?
#   PNGwriter_DEFINITIONS     - Compiler definitions you should add with
#                               add_definitions(${PNGwriter_DEFINITIONS})
#


###############################################################################
# Copyright 2014-2015 Axel Huebl
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

cmake_minimum_required(VERSION 2.8.5)


###############################################################################
# PNGwriter
###############################################################################

# we start by assuming we found PNGwriter and falsify it if some
# dependencies are missing (or if we did not find PNGwriter at all)
set(PNGwriter_FOUND TRUE)

# find PNGwriter install ######################################################
#
find_path(PNGwriter_ROOT_DIR
    NAMES include/pngwriter.h
    PATHS ENV PNGWRITER_ROOT
    DOC "PNGwriter ROOT location")

if(NOT PNGwriter_ROOT_DIR)
    set(PNGwriter_FOUND FALSE)
endif(NOT PNGwriter_ROOT_DIR)

# option: prefer static libs ##################################################
#
if(PNGwriter_USE_STATIC_LIBS)
    # carefully: we have to restore the original path in the end
    set(_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
    set(CMAKE_FIND_LIBRARY_SUFFIXES .a .so)
endif()

# find libpng install #########################################################
#
find_package(PNG 1.2.9)

if(NOT PNG_FOUND)
    set(PNGwriter_FOUND FALSE)
endif(NOT PNG_FOUND)


if(PNGwriter_FOUND)
    # PNGwriter headers #######################################################
    #
    list(APPEND PNGwriter_INCLUDE_DIRS ${PNGwriter_ROOT_DIR}/include)
    list(APPEND PNGwriter_INCLUDE_DIRS ${PNG_INCLUDE_DIRS})

    # PNGwriter libraries #####################################################
    #
    find_library(PNGwriter_LIBRARIES
        NAMES pngwriter
        PATHS ${PNGwriter_ROOT_DIR}/lib)

    #  libpng
    list(APPEND PNGwriter_LIBRARIES ${PNG_LIBRARIES})

    # PNGwriter definitions ###################################################
    #
    list(APPEND PNGwriter_DEFINITIONS ${PNG_DEFINITIONS})

    #   freetype support enabled?
    #   (assumes: environment did not change since install)
    include(FindFreetype)
    if(FREETYPE_FOUND)
        list(APPEND PNGwriter_INCLUDE_DIRS ${FREETYPE_INCLUDE_DIRS})
        list(APPEND PNGwriter_LIBRARIES ${FREETYPE_LIBRARIES})

    else(FREETYPE_FOUND)
        # this flag is important for pngwriter headers, see
        # https://github.com/ax3l/pngwriter/issues/7
        list(APPEND PNGwriter_DEFINITIONS "-DNO_FREETYPE")
    endif(FREETYPE_FOUND)

    # version string ##########################################################
    #
    set(PNGwriter_VERSION "0.5.4")

else(PNGwriter_FOUND)
    message(STATUS "Can NOT find PNGwriter - set PNGWRITER_ROOT")
endif(PNGwriter_FOUND)


# restore defaults of library suffixes ####################################
#
if(PNGwriter_USE_STATIC_LIBS)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()


# unset checked variables if not found ########################################
#
if(NOT PNGwriter_FOUND)
    unset(PNGwriter_INCLUDE_DIRS)
    unset(PNGwriter_LIBRARIES)
    unset(PNGwriter_HAS_FREETYPE)
endif(NOT PNGwriter_FOUND)


###############################################################################
# FindPackage Options
###############################################################################

# handles the REQUIRED, QUIET and version-related arguments for find_package
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PNGwriter
    REQUIRED_VARS PNGwriter_LIBRARIES PNGwriter_INCLUDE_DIRS
    VERSION_VAR PNGwriter_VERSION
)
