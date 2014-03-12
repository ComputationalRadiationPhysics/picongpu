# - Find ADIOS library, routines for scientific, parallel IO
#   https://www.olcf.ornl.gov/center-projects/adios/
#
# Module that finds the includes and libraries for a working ADIOS install.
# This module invokes the `adios_config` script that should be installed with
# the other ADIOS tools.
#
# To provide a hint to the module where to find the ADIOS installation,
# set the ADIOS_ROOT environment variable.
#
# This module will define the following variables:
#   ADIOS_INCLUDE_DIRS   - Include directories for the ADIOS headers.
#   ADIOS_LIBRARIES      - ADIOS libraries.
#   ADIOS_FOUND          - TRUE if FindADIOS found a working install
#   ADIOS_VERSION        - Version in format Major.Minor.Patch
#
# Not used for now:
#   ADIOS_DEFINITIONS    - Compiler definitions you should add with
#                          add_definitons(${ADIOS_DEFINITIONS})
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

# we start by assuming we found ADIOS and fasily if some
# dependencies are missing (or if we did not find ADIOS at all)
set(ADIOS_FOUND TRUE)

# check at ADIOS_ROOT
find_path(ADIOS_ROOT_DIR
  NAMES include/adios.h lib/libadios.so
  PATHS ENV ADIOS_ROOT
  DOC "ADIOS ROOT location"
)

# we found something in ADIOS_ROOT
if(ADIOS_ROOT_DIR)
    message(STATUS "Found ADIOS: "${ADIOS_ROOT_DIR})

    list(APPEND ADIOS_INCLUDE_DIRS ${ADIOS_ROOT_DIR}/include)
    list(APPEND ADIOS_LIBRARIES ${ADIOS_ROOT_DIR}/lib/libadios.a)

    # find mxml installation
    find_path(MXML_ROOT_DIR
      NAMES include/mxml.h lib/libmxml.so
      PATHS ENV MXML_ROOT
      DOC "MXML ROOT location (ADIOS dependency)"
    )

    if(MXML_ROOT_DIR)
        message(STATUS "Found MXML: "${MXML_ROOT_DIR})

        list(APPEND ADIOS_INCLUDE_DIRS ${MXML_ROOT_DIR}/include)
        list(APPEND ADIOS_LIBRARIES ${MXML_ROOT_DIR}/lib/libmxml.a)

    else(MXML_ROOT_DIR)
        set(ADIOS_FOUND FALSE)
        message(STATUS "Could NOT find MXML (ADIOS dependency)")
    endif(MXML_ROOT_DIR)

    # check for further dependencies (right now Dataspaces)
    execute_process(COMMAND adios_config -l
                    OUTPUT_VARIABLE ADIOS_LINKFLAGS)
    message(STATUS "Additional linker flags for ADIOS: ${ADIOS_LINKFLAGS}")

    # find dataspaces installation (if compiled in)
    if("${ADIOS_LINKFLAGS}" MATCHES "ldspaces")
        find_path(DATASPACES_ROOT_DIR
          NAMES include/dataspaces.h lib/libdspaces.a
          PATHS ENV DATASPACES_ROOT
          DOC "DATASPACES ROOT location (ADIOS dependency)"
        )

       if(DATASPACES_ROOT_DIR)
            message(STATUS "Found Dataspaces: "${DATASPACES_ROOT_DIR})

            list(APPEND ADIOS_INCLUDE_DIRS ${DATASPACES_ROOT_DIR}/include)
            list(APPEND ADIOS_LIBRARIES ${DATASPACES_ROOT_DIR}/lib/libdspaces.a)
            list(APPEND ADIOS_LIBRARIES ${DATASPACES_ROOT_DIR}/lib/libdscommon.a)
            list(APPEND ADIOS_LIBRARIES ${DATASPACES_ROOT_DIR}/lib/libdart.a)

        else(DATASPACES_ROOT_DIR)
            set(ADIOS_FOUND FALSE)
            message(STATUS "Could NOT find DATASPACES (ADIOS dependency)")
        endif(DATASPACES_ROOT_DIR)
    endif()

    # add the version string
    execute_process(COMMAND adios_config -v
                    OUTPUT_VARIABLE ADIOS_VERSION)
    # trim trailing newlines
    string(REGEX REPLACE "(\r?\n)+$" "" ADIOS_VERSION "${ADIOS_VERSION}")

else(ADIOS_ROOT_DIR)
    set(ADIOS_FOUND FALSE)
    unset(ADIOS_INCLUDE_DIRS)
    unset(ADIOS_LIBRARIES)
endif(ADIOS_ROOT_DIR)


################################################################################
# FindPackage Options
################################################################################

# handles the REQUIRED, QUIET and version-related arguments for find_package
find_package_handle_standard_args(ADIOS
    REQUIRED_VARS ADIOS_LIBRARIES ADIOS_INCLUDE_DIRS
    VERSION_VAR ADIOS_VERSION
)
