#
# Copyright 2016-2021 Rene Widera, Benjamin Worpitz, Simeon Ehrig
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

cmake_minimum_required(VERSION 3.22.0)

################################################################################
# Setup project information
################################################################################

# Find cupla version.
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/include/cupla/version.hpp" CUPLA_VERSION_MAJOR_HPP REGEX "#define CUPLA_VERSION_MAJOR ")
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/include/cupla/version.hpp" CUPLA_VERSION_MINOR_HPP REGEX "#define CUPLA_VERSION_MINOR ")
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/include/cupla/version.hpp" CUPLA_VERSION_PATCH_HPP REGEX "#define CUPLA_VERSION_PATCH ")

string(REGEX MATCH "([0-9]+)" CUPLA_VERSION_MAJOR  ${CUPLA_VERSION_MAJOR_HPP})
string(REGEX MATCH "([0-9]+)" CUPLA_VERSION_MINOR  ${CUPLA_VERSION_MINOR_HPP})
string(REGEX MATCH "([0-9]+)" CUPLA_VERSION_PATCH  ${CUPLA_VERSION_PATCH_HPP})

set(PACKAGE_VERSION "${CUPLA_VERSION_MAJOR}.${CUPLA_VERSION_MINOR}.${CUPLA_VERSION_PATCH}")


project(cupla VERSION      ${CUPLA_VERSION_MAJOR}.${CUPLA_VERSION_MINOR}.${CUPLA_VERSION_PATCH}
              DESCRIPTION  "cupla is a simple CUDA like user interface for the platform independent parallel kernel acceleration library alpaka."
              HOMEPAGE_URL "https://github.com/alpaka-group/cupla"
              LANGUAGES    CXX)
include(GNUInstallDirs)

################################################################################
# cupla options
################################################################################

option(CUPLA_STREAM_ASYNC_ENABLE "Enable asynchronous streams" ON)
option(cupla_BUILD_EXAMPLES "Build examples" OFF)

################################################################################
# setup alpaka
################################################################################

# the min and max. supported alpaka version is also copied to the cuplaConfig.cmake
set(_CUPLA_MIN_ALPAKA_VERSION 1.0.0)
set(_CUPLA_MAX_ALPAKA_VERSION 1.0.0)

# do not search for alpaka if it already exists
# for example, a project that includes alpaka via add_subdirectory before including cupla via add_subdirectory
if(NOT TARGET alpaka::alpaka)
  # the alpaka provider for the internal alpaka is only available,
  # if cupla is used via add_subdirectory in another project
  # or examples are build
  if(cupla_BUILD_EXAMPLES OR (NOT ${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME}))
    set(cupla_ALPAKA_PROVIDER "internal" CACHE STRING "Select which alpaka is used")
    set_property(CACHE cupla_ALPAKA_PROVIDER PROPERTY STRINGS "internal;external")
    mark_as_advanced(cupla_ALPAKA_PROVIDER)

    if(${cupla_ALPAKA_PROVIDER} STREQUAL "internal")
      set(alpaka_BUILD_EXAMPLES OFF)
      set(BUILD_TESTING OFF)
      add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/alpaka)
    else()
      find_package(alpaka ${_CUPLA_MAX_ALPAKA_VERSION} HINTS $ENV{ALPAKA_ROOT})
      if(NOT TARGET alpaka::alpaka)
        message(STATUS "Could not find alpaka ${_CUPLA_MAX_ALPAKA_VERSION}. Now searching for alpaka ${_CUPLA_MIN_ALPAKA_VERSION}")
        find_package(alpaka ${_CUPLA_MIN_ALPAKA_VERSION} REQUIRED HINTS $ENV{ALPAKA_ROOT})
      endif()
      if(alpaka_VERSION VERSION_GREATER _CUPLA_MAX_ALPAKA_VERSION)
        message(WARNING "Unsupported alpaka version ${alpaka_VERSION}. "
          "Supported versions [${_CUPLA_MIN_ALPAKA_VERSION},${_CUPLA_MAX_ALPAKA_VERSION}].")
      endif()
    endif()

    if(NOT TARGET alpaka::alpaka)
      message(FATAL_ERROR "Required cupla dependency alpaka could not be found!")
    endif()
  endif()
endif()

################################################################################
# cupla Target.
################################################################################

# create cupla target only if the cupla is used via add_subdirectory
# or examples are build
# for the explanation please have a look in the cuplaConfig.cmake.in
if(cupla_BUILD_EXAMPLES OR (NOT ${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME}))
  include("${CMAKE_CURRENT_LIST_DIR}/cmake/addExecutable.cmake")
  include("${CMAKE_CURRENT_LIST_DIR}/cmake/cuplaTargetHelper.cmake")

  # export HIP_HIPCC_FLAGS to the parent scope else the variable is not visible
  # for application files
  if(HIP_HIPCC_FLAGS)
    set(HIP_HIPCC_FLAGS ${HIP_HIPCC_FLAGS} PARENT_SCOPE)
  endif()

  createCuplaTarget(${PROJECT_NAME}
    ${PROJECT_SOURCE_DIR}/include # include directory path
    ${PROJECT_SOURCE_DIR}/src # src directory path
    )
endif()


################################################################################
# add examples
################################################################################

if(cupla_BUILD_EXAMPLES)
  add_subdirectory(example/)
endif()

################################################################################
# install cupla
################################################################################

if(${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME})
  include(CMakePackageConfigHelpers)

  set(_CUPLA_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
  set(_CUPLA_SOURCE_CMAKEDIR "${PROJECT_SOURCE_DIR}/cmake/")

  write_basic_package_version_file("${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

  configure_package_config_file(
    "${_CUPLA_SOURCE_CMAKEDIR}/${PROJECT_NAME}Config.cmake.in"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION ${_CUPLA_INSTALL_CMAKEDIR}
    PATH_VARS _CUPLA_MIN_ALPAKA_VERSION _CUPLA_MAX_ALPAKA_VERSION)

  install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION ${_CUPLA_INSTALL_CMAKEDIR})

  install(FILES "${_CUPLA_SOURCE_CMAKEDIR}/addExecutable.cmake"
    "${_CUPLA_SOURCE_CMAKEDIR}/cuplaTargetHelper.cmake"
    DESTINATION ${_CUPLA_INSTALL_CMAKEDIR})

  install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)
  # copy source files instead compiled library
  # this is necessary because some functions use the ACC as a template parameter,
  # but the ACC is not defined at the install time of cupla
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/ DESTINATION src/cupla)
endif()
