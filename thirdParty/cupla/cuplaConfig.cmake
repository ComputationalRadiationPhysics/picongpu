#
# Copyright 2016 Rene Widera, Benjamin Worpitz
#
# This file is part of cupla.
#
# cupla is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cupla is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with cupla.
# If not, see <http://www.gnu.org/licenses/>.
#

################################################################################
# Required cmake version.
################################################################################

cmake_minimum_required(VERSION 3.15.0)

################################################################################
# CMake policies
#
# Search in <PackageName>_ROOT:
#   https://cmake.org/cmake/help/v3.12/policy/CMP0074.html
################################################################################

if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

################################################################################
# cupla
################################################################################

# Return values.
unset(cupla_FOUND)
unset(cupla_VERSION)
unset(cupla_COMPILE_OPTIONS)
unset(cupla_COMPILE_DEFINITIONS)
unset(cupla_DEFINITIONS)
unset(cupla_INCLUDE_DIR)
unset(cupla_INCLUDE_DIRS)
unset(cupla_SOURCE_DIR)
unset(cupla_SOURCE_DIRS)
unset(cupla_LIBRARY)
unset(cupla_LIBRARIES)
unset(cupla_SOURCE_FILES)

# Internal usage.
unset(_cupla_FOUND)
unset(_cupla_VERSION)
unset(_cupla_COMPILE_OPTIONS_PUBLIC)
unset(_cupla_COMPILE_DEFINITIONS_PUBLIC)
unset(_cupla_INCLUDE_DIR)
unset(_cupla_INCLUDE_DIRECTORIES_PUBLIC)
unset(_cupla_SOURCE_DIR)
unset(_cupla_SOURCE_DIRECTORIES_PUBLIC)
unset(_cupla_LINK_LIBRARIES_PUBLIC)
unset(_cupla_FILES_HEADER)
unset(_cupla_FILES_SOURCE)
unset(_cupla_FILES_OTHER)
unset(_cupla_ADD_EXECUTABLE_FILE)

################################################################################
# Directory of this file.
################################################################################
set(_cupla_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})

# Normalize the path (e.g. remove ../)
get_filename_component(_cupla_ROOT_DIR "${_cupla_ROOT_DIR}" ABSOLUTE)

# Add cupla_ADD_EXECUTABLE function.
set(_cupla_ADD_EXECUTABLE_FILE "${_cupla_ROOT_DIR}/cmake/addExecutable.cmake")
include("${_cupla_ADD_EXECUTABLE_FILE}")

################################################################################
# Set found to true initially and set it on false if a required dependency is missing.
################################################################################
set(_cupla_FOUND TRUE)

################################################################################
# Common.
################################################################################
# own modules for find_packages
list(APPEND CMAKE_MODULE_PATH "${_cupla_ROOT_DIR}")
#list(APPEND CMAKE_MODULE_PATH "$ENV{ALPAKA_ROOT}")


################################################################################
# Find alpaka
# NOTE: Do this first, because it declares `list_add_prefix` and `append_recursive_files_add_to_src_group` used later on.
################################################################################

# disable all accelerators by default
OPTION(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE "Enable the serial CPU accelerator" OFF)
OPTION(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE "Enable the threads CPU block thread accelerator" OFF)
OPTION(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE "Enable the fibers CPU block thread accelerator" OFF)
OPTION(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE "Enable the TBB CPU grid block back-end" OFF)
OPTION(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE "Enable the OpenMP 2.0 CPU grid block accelerator" OFF)
OPTION(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE "Enable the OpenMP 2.0 CPU block thread accelerator" OFF)
OPTION(ALPAKA_ACC_CPU_BT_OMP4_ENABLE "Enable the OpenMP 4.0 CPU block and block thread accelerator" OFF)
OPTION(ALPAKA_ACC_GPU_CUDA_ENABLE "Enable the CUDA GPU accelerator" OFF)
OPTION(ALPAKA_ACC_GPU_HIP_ENABLE "Enable the HIP back-end (all other back-ends must be disabled)" OFF)

set(cupla_ALPAKA_PROVIDER "intern" CACHE STRING "Select which alpaka is used")
set_property(CACHE cupla_ALPAKA_PROVIDER PROPERTY STRINGS "intern;extern")
mark_as_advanced(cupla_ALPAKA_PROVIDER)

if(${cupla_ALPAKA_PROVIDER} STREQUAL "intern")
    set(alpaka_BUILD_EXAMPLES OFF)
    set(BUILD_TESTING OFF)
    add_subdirectory(${_cupla_ROOT_DIR}/alpaka ${CMAKE_BINARY_DIR}/alpaka)
else()
    find_package(alpaka HINTS $ENV{ALPAKA_ROOT})
endif()

if(NOT TARGET alpaka::alpaka)
    message(WARNING "Required cupla dependency alpaka could not be found!")
    set(_cupla_FOUND FALSE)
endif()

################################################################################
# Compiler settings.
################################################################################
if(MSVC)
    # Empty append to define it if it does not already exist.
    list(APPEND _cupla_COMPILE_OPTIONS_PUBLIC)
else()
    # GNU
    if(CMAKE_COMPILER_IS_GNUCXX)
        list(APPEND _cupla_COMPILE_OPTIONS_PUBLIC "-Wall")
        list(APPEND _cupla_COMPILE_OPTIONS_PUBLIC "-Wextra")
        list(APPEND _cupla_COMPILE_OPTIONS_PUBLIC "-Wno-unknown-pragmas")
        list(APPEND _cupla_COMPILE_OPTIONS_PUBLIC "-Wno-unused-parameter")
        list(APPEND _cupla_COMPILE_OPTIONS_PUBLIC "-Wno-unused-local-typedefs")
        list(APPEND _cupla_COMPILE_OPTIONS_PUBLIC "-Wno-attributes")
        list(APPEND _cupla_COMPILE_OPTIONS_PUBLIC "-Wno-reorder")
        list(APPEND _cupla_COMPILE_OPTIONS_PUBLIC "-Wno-sign-compare")
    # ICC
    elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
        list(APPEND _cupla_COMPILE_OPTIONS_PUBLIC "-Wall")
    # PGI
    elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "PGI")
        list(APPEND _cupla_COMPILE_OPTIONS_PUBLIC "-Minform=inform")
    endif()
endif()


################################################################################
# cupla.
################################################################################

option(CUPLA_STREAM_ASYNC_ENABLE "Enable asynchronous streams" ON)
if(CUPLA_STREAM_ASYNC_ENABLE)
    list(APPEND _cupla_COMPILE_DEFINITIONS_PUBLIC "CUPLA_STREAM_ASYNC_ENABLED=1")
else()
    list(APPEND _cupla_COMPILE_DEFINITIONS_PUBLIC "CUPLA_STREAM_ASYNC_ENABLED=0")
endif()

set(_cupla_INCLUDE_DIR "${_cupla_ROOT_DIR}/include")
list(APPEND _cupla_INCLUDE_DIRECTORIES_PUBLIC ${_cupla_INCLUDE_DIR})
set(_cupla_SUFFIXED_INCLUDE_DIR "${_cupla_INCLUDE_DIR}")

set(_cupla_SOURCE_DIR "${_cupla_ROOT_DIR}/src")
list(APPEND _cupla_SOURCE_DIRECTORIES_PUBLIC ${_cupla_SOURCE_DIR})
set(_cupla_SUFFIXED_SOURCE_DIR "${_cupla_SOURCE_DIR}")

set(_cupla_LINK_LIBRARY)
list(APPEND _cupla_LINK_LIBRARIES_PUBLIC ${_cupla_LINK_LIBRARY})

set(_cupla_FILES_OTHER "${_cupla_ROOT_DIR}/Findcupla.cmake" "${_cupla_ROOT_DIR}/cuplaConfig.cmake")

# Add all the include files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_src_group("${_cupla_SUFFIXED_INCLUDE_DIR}" "${_cupla_SUFFIXED_INCLUDE_DIR}" "hpp" _cupla_FILES_HEADER)
append_recursive_files_add_to_src_group("${_cupla_SUFFIXED_INCLUDE_DIR}" "${_cupla_SUFFIXED_INCLUDE_DIR}" "h" _cupla_FILES_HEADER)

append_recursive_files_add_to_src_group("${_cupla_SUFFIXED_SOURCE_DIR}" "${_cupla_SUFFIXED_SOURCE_DIR}" "cpp" _cupla_FILES_SOURCE)


################################################################################
# cupla Target.
################################################################################

alpaka_add_library(
    "cupla"
    ${_cupla_FILES_HEADER} ${_cupla_FILES_OTHER} ${_cupla_FILES_SOURCE})

# Even if there are no sources CMAKE has to know the language.
set_target_properties("cupla" PROPERTIES LINKER_LANGUAGE CXX)

# Compile options.
message(STATUS "_cupla_COMPILE_OPTIONS_PUBLIC: ${_cupla_COMPILE_OPTIONS_PUBLIC}")
list(
    LENGTH
    _cupla_COMPILE_OPTIONS_PUBLIC
    _cupla_COMPILE_OPTIONS_PUBLIC_LENGTH)
if("${_cupla_COMPILE_OPTIONS_PUBLIC_LENGTH}")
    TARGET_COMPILE_OPTIONS(
        "cupla"
        PUBLIC ${_cupla_COMPILE_OPTIONS_PUBLIC})
endif()

# Compile definitions.
message(STATUS "_cupla_COMPILE_DEFINITIONS_PUBLIC: ${_cupla_COMPILE_DEFINITIONS_PUBLIC}")
list(
    LENGTH
    _cupla_COMPILE_DEFINITIONS_PUBLIC
    _cupla_COMPILE_DEFINITIONS_PUBLIC_LENGTH)
if("${_cupla_COMPILE_DEFINITIONS_PUBLIC_LENGTH}")
    TARGET_COMPILE_DEFINITIONS(
        "cupla"
        PUBLIC ${_cupla_COMPILE_DEFINITIONS_PUBLIC})
endif()

# Include directories.
message(STATUS "_cupla_INCLUDE_DIRECTORIES_PUBLIC: ${_cupla_INCLUDE_DIRECTORIES_PUBLIC}")
list(
    LENGTH
    _cupla_INCLUDE_DIRECTORIES_PUBLIC
    _cupla_INCLUDE_DIRECTORIES_PUBLIC_LENGTH)
if("${_cupla_INCLUDE_DIRECTORIES_PUBLIC_LENGTH}")
    TARGET_INCLUDE_DIRECTORIES(
        "cupla"
        PUBLIC ${_cupla_INCLUDE_DIRECTORIES_PUBLIC})
endif()

# Link libraries.
message(STATUS "_cupla_LINK_LIBRARIES_PUBLIC: ${_cupla_LINK_LIBRARIES_PUBLIC}")

target_link_libraries(
    "cupla"
    PUBLIC alpaka::alpaka ${_cupla_LINK_LIBRARIES_PUBLIC})

################################################################################
# Find cupla version.
################################################################################
# Please also update the version in `include/cupla/version.hpp`
set(_cupla_VERSION "0.2.0")

################################################################################
# Set return values.
################################################################################
set(cupla_VERSION ${_cupla_VERSION})
set(cupla_COMPILE_OPTIONS ${_cupla_COMPILE_OPTIONS_PUBLIC})
set(cupla_COMPILE_DEFINITIONS ${_cupla_COMPILE_DEFINITIONS_PUBLIC})
# Add '-D' to the definitions
set(cupla_DEFINITIONS ${_cupla_COMPILE_DEFINITIONS_PUBLIC})
list_add_prefix("-D" cupla_DEFINITIONS)
# Add the compile options to the definitions.
list(APPEND cupla_DEFINITIONS ${_cupla_COMPILE_OPTIONS_PUBLIC})
set(cupla_INCLUDE_DIR ${_cupla_INCLUDE_DIR})
set(cupla_INCLUDE_DIRS ${_cupla_INCLUDE_DIRECTORIES_PUBLIC})
set(cupla_LIBRARY ${_cupla_LINK_LIBRARY})
set(cupla_LIBRARIES ${_cupla_LINK_LIBRARIES_PUBLIC})
set(cupla_SOURCE_FILES ${_cupla_FILES_SOURCE})

# Unset already set variables if not found.
if(NOT _cupla_FOUND)
    unset(cupla_FOUND)
    unset(cupla_VERSION)
    unset(cupla_COMPILE_OPTIONS)
    unset(cupla_COMPILE_DEFINITIONS)
    unset(cupla_DEFINITIONS)
    unset(cupla_INCLUDE_DIR)
    unset(cupla_INCLUDE_DIRS)
    unset(cupla_SOURCE_DIR)
    unset(cupla_SOURCE_DIRS)
    unset(cupla_LIBRARY)
    unset(cupla_LIBRARIES)
    unset(cupla_SOURCE_FILES)

    unset(_cupla_FOUND)
    unset(_cupla_COMPILE_OPTIONS_PUBLIC)
    unset(_cupla_COMPILE_DEFINITIONS_PUBLIC)
    unset(_cupla_INCLUDE_DIR)
    unset(_cupla_INCLUDE_DIRECTORIES_PUBLIC)
    unset(_cupla_SOURCE_DIR)
    unset(_cupla_SOURCE_DIRECTORIES_PUBLIC)
    unset(_cupla_LINK_LIBRARY)
    unset(_cupla_LINK_LIBRARIES_PUBLIC)
    unset(_cupla_FILES_HEADER)
    unset(_cupla_FILES_SOURCE)
    unset(_cupla_FILES_OTHER)
    unset(_cupla_VERSION)
    unset(_cupla_ADD_EXECUTABLE_FILE)
else()
    # Make internal variables advanced options in the GUI.
    MARK_AS_ADVANCED(
        cupla_INCLUDE_DIR
        cupla_LIBRARY
        _cupla_FOUND
        _cupla_COMPILE_OPTIONS_PUBLIC
        _cupla_COMPILE_DEFINITIONS_PUBLIC
        _cupla_INCLUDE_DIR
        _cupla_INCLUDE_DIRECTORIES_PUBLIC
        _cupla_LINK_LIBRARY
        _cupla_LINK_LIBRARIES_PUBLIC
        _cupla_FILES_HEADER
        _cupla_FILES_SOURCE
        _cupla_FILES_OTHER
        _cupla_VERSION
        _cupla_ADD_EXECUTABLE_FILE
    )
endif()

###############################################################################
# FindPackage options
###############################################################################

# Handles the REQUIRED, QUIET and version-related arguments for find_package.
# NOTE: We do not check for cupla_LIBRARIES and cupla_DEFINITIONS because they can be empty.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    "cupla"
    FOUND_VAR cupla_FOUND
    REQUIRED_VARS cupla_INCLUDE_DIR
    VERSION_VAR cupla_VERSION)
