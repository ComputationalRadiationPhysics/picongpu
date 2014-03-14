# - Find ADIOS library, routines for scientific, parallel IO
#   https://www.olcf.ornl.gov/center-projects/adios/
#
# Use this module by invoking find_package with the form:
#   find_package(ADIOS
#     [version] [EXACT]     # Minimum or EXACT version, e.g. 1.6.0
#     [REQUIRED]            # Fail with an error if ADIOS or a required
#                           #   component is not found
#     [QUIET]               # ...
#     [COMPONENTS <...>]    # Compiled in components, ignored
#   )
#
# Module that finds the includes and libraries for a working ADIOS install.
# This module invokes the `adios_config` script that should be installed with
# the other ADIOS tools.
#
# To provide a hint to the module where to find the ADIOS installation,
# set the ADIOS_ROOT environment variable.
#
# If this variable is not set, make sure that at least the according `bin/`
# directory of ADIOS is in your PATH environment variable.
#
# Set the following CMake variables BEFORE calling find_packages to
# influence this module:
#   ADIOS_USE_STATIC_LIBS - Set to ON to force the use of static
#                           libraries.  Default: OFF
#
# This module will define the following variables:
#   ADIOS_INCLUDE_DIRS    - Include directories for the ADIOS headers.
#   ADIOS_LIBRARIES       - ADIOS libraries.
#   ADIOS_FOUND           - TRUE if FindADIOS found a working install
#   ADIOS_VERSION         - Version in format Major.Minor.Patch
#
# Not used for now:
#   ADIOS_DEFINITIONS     - Compiler definitions you should add with
#                           add_definitions(${ADIOS_DEFINITIONS})
#


################################################################################
# Copyright 2014 Axel Huebl, Felix Schmitt                          
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
################################################################################


################################################################################
# Required cmake version
################################################################################

cmake_minimum_required(VERSION 2.8.5)


################################################################################
# ADIOS
################################################################################

# we start by assuming we found ADIOS and falsify it if some
# dependencies are missing (or if we did not find ADIOS at all)
set(ADIOS_FOUND TRUE)


# find `adios_config` program #################################################
#   check the ADIOS_ROOT hint and the normal PATH
find_file(ADIOS_CONFIG
    NAME adios_config
    PATHS $ENV{ADIOS_ROOT}/bin $ENV{PATH})

if(ADIOS_CONFIG)
    message(STATUS "Found 'adios_config': ${ADIOS_CONFIG}")
else(ADIOS_CONFIG)
    set(ADIOS_FOUND FALSE)
    message(STATUS "Can NOT find 'adios_config' - set ADIOS_ROOT or check your PATH")
endif(ADIOS_CONFIG)


# check `adios_config` program ################################################
if(ADIOS_FOUND)
    execute_process(COMMAND ${ADIOS_CONFIG} -l
                    OUTPUT_VARIABLE ADIOS_LINKFLAGS
                    RESULT_VARIABLE ADIOS_CONFIG_RETURN)
    if(NOT ADIOS_CONFIG_RETURN EQUAL 0)
        set(ADIOS_FOUND FALSE)
        message(STATUS "Can NOT execute 'adios_config' - check file permissions")
    else()
        # trim trailing newlines
        string(REGEX REPLACE "(\r?\n)+$" "" ADIOS_LINKFLAGS "${ADIOS_LINKFLAGS}")
    endif()

    # find ADIOS_ROOT_DIR
    execute_process(COMMAND ${ADIOS_CONFIG} -d
                    OUTPUT_VARIABLE ADIOS_ROOT_DIR)
    # trim trailing newlines
    string(REGEX REPLACE "(\r?\n)+$" "" ADIOS_ROOT_DIR "${ADIOS_ROOT_DIR}")
    if(NOT IS_DIRECTORY "${ADIOS_ROOT_DIR}")
        set(ADIOS_FOUND FALSE)
        message(STATUS "The directory provided by 'adios_config -d' does not exist: ${ADIOS_ROOT_DIR}")
    endif()
endif(ADIOS_FOUND)


# option: use only static libs ################################################
if(ADIOS_USE_STATIC_LIBS)
    # carfully: we have to restore the original path in the end
    set(_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
    set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
endif()


# we found something in ADIOS_ROOT_DIR and adios_config works #################
if(ADIOS_FOUND)
    # ADIOS headers
    list(APPEND ADIOS_INCLUDE_DIRS ${ADIOS_ROOT_DIR}/include)

    # check for compiled in dependencies
    message(STATUS "ADIOS linker flags (unparsed): ${ADIOS_LINKFLAGS}")

    # find all library paths -L
    #   note: this can cause trouble if some libs are specified twice from
    #         different sources (quite unlikely)
    #         http://www.cmake.org/pipermail/cmake/2008-November/025128.html
    set(ADIOS_LIBRARY_DIRS "")
    string(REGEX MATCHALL "-L([A-Za-z_0-9/\\.-]+)" _ADIOS_LIBDIRS "${ADIOS_LINKFLAGS}")
    foreach(_LIBDIR ${_ADIOS_LIBDIRS})
        string(REPLACE "-L" "" _LIBDIR ${_LIBDIR})
        list(APPEND ADIOS_LIBRARY_DIRS ${_LIBDIR})
    endforeach()
    # we could append ${CMAKE_PREFIX_PATH} now but that is not really necessary

    #message(STATUS "ADIOS DIRS to look for libs: ${ADIOS_LIBRARY_DIRS}")

    # parse all -lname libraries and find an absolute path for them
    string(REGEX MATCHALL "-l([A-Za-z_0-9\\.-]+)" _ADIOS_LIBS "${ADIOS_LINKFLAGS}")

    foreach(_LIB ${_ADIOS_LIBS})
        string(REPLACE "-l" "" _LIB ${_LIB})

        # find static lib: absolute path in -L then default
        find_library(_LIB_DIR NAMES ${_LIB} PATHS ${ADIOS_LIBRARY_DIRS})

        # found?
        if(_LIB_DIR)
            message(STATUS "Found ${_LIB} in ${_LIB_DIR}")
            list(APPEND ADIOS_LIBRARIES "${_LIB_DIR}")
        else(_LIB_DIR)
            set(ADIOS_FOUND FALSE)
            message(STATUS "ADIOS: Could NOT find library '${_LIB}'")
        endif(_LIB_DIR)

        # clean cached var
        unset(_LIB_DIR CACHE)
        unset(_LIB_DIR)
    endforeach()

    # simplify lists and check for missing components (not implemented)
    set(ADIOS_MISSING_COMPONENTS "")
    foreach(COMPONENT ${ADIOS_FIND_COMPONENTS})
        string(TOUPPER ${COMPONENT} COMPONENT)
        list(APPEND ADIOS_MISSING_COMPONENTS ${COMPONENT})
    endforeach()
    #message(STATUS "ADIOS required components: ${ADIOS_FIND_COMPONENTS}")

    # add the version string
    execute_process(COMMAND ${ADIOS_CONFIG} -v
                    OUTPUT_VARIABLE ADIOS_VERSION)
    # trim trailing newlines
    string(REGEX REPLACE "(\r?\n)+$" "" ADIOS_VERSION "${ADIOS_VERSION}")

endif(ADIOS_FOUND)

# unset checked variables if not found
if(NOT ADIOS_FOUND)
    unset(ADIOS_INCLUDE_DIRS)
    unset(ADIOS_LIBRARIES)
endif(NOT ADIOS_FOUND)


# restore CMAKE_FIND_LIBRARY_SUFFIXES if manipulated by this module ###########
if(ADIOS_USE_STATIC_LIBS)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()


###############################################################################
# FindPackage Options
###############################################################################

# handles the REQUIRED, QUIET and version-related arguments for find_package
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ADIOS
    REQUIRED_VARS ADIOS_LIBRARIES ADIOS_INCLUDE_DIRS
    VERSION_VAR ADIOS_VERSION
)
