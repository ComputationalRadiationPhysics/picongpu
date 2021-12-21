# Copyright 2013-2021 Rene Widera, Axel Huebl
#
# This file is part of PMacc.
#
# PMacc is free software: you can redistribute it and/or modify
# it under the terms of either the GNU General Public License or
# the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PMacc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License and the GNU Lesser General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# and the GNU Lesser General Public License along with PMacc.
# If not, see <http://www.gnu.org/licenses/>.
#

################################################################################
# Required cmake version
################################################################################

cmake_minimum_required(VERSION 3.15.0)


################################################################################
# Project
################################################################################

project(GameOfLife)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "install prefix" FORCE)
endif()

# set helper pathes to find libraries and packages
# Add specific hints
list(APPEND CMAKE_PREFIX_PATH "$ENV{MPI_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{BOOST_ROOT}")
# Add from environment after specific env vars
list(APPEND CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH}")


################################################################################
# CMake policies
#
# Search in <PackageName>_ROOT:
#   https://cmake.org/cmake/help/v3.12/policy/CMP0074.html
################################################################################

if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()


###############################################################################
# Language Flags
###############################################################################

# enforce C++14
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 14)


################################################################################
# Find PMacc
################################################################################

find_package(PMacc REQUIRED CONFIG PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/pmacc")
include_directories(SYSTEM ${PMacc_INCLUDE_DIRS})
set(LIBS ${LIBS} ${PMacc_LIBRARIES})
add_definitions(${PMacc_DEFINITIONS})


################################################################################
# Find MPI
################################################################################

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
set(LIBS ${LIBS} ${MPI_C_LIBRARIES})


################################################################################
# Build type (debug, release)
################################################################################

option(GOL_RELEASE "Build release version, disables all runtime asserts" OFF)
if(GOL_RELEASE)
    message(STATUS "Release version")
    set(CMAKE_BUILD_TYPE Release)
    add_definitions(-DNDEBUG)
else(GOL_RELEASE)
    message(STATUS "Debug version")
    set(CMAKE_BUILD_TYPE Debug)
endif(GOL_RELEASE)


###############################################################################
# Find Boost
###############################################################################

find_package(Boost 1.65.1 REQUIRED COMPONENTS program_options)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Boost_LIBRARIES})


################################################################################
# PNGwriter
################################################################################

# find PNGwriter installation
find_package(PNGwriter 0.7.0 REQUIRED CONFIG)

if(PNGwriter_FOUND)
    list(APPEND PNGwriter_DEFINITIONS "-DGOL_ENABLE_PNG=1")
    set(LIBS ${LIBS} PNGwriter::PNGwriter)
endif(PNGwriter_FOUND)

################################################################################
# Math
################################################################################

if(NOT WIN32)
    # automatically added on windows
    set(LIBS ${LIBS} m)
endif()


################################################################################
# Warnings
################################################################################

# GNU
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
    # new warning in gcc 4.8 (flag ignored in previous version)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-local-typedefs")
# ICC
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
# PGI
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Minform=inform")
endif()


################################################################################
# Compile & Link GoL
################################################################################

include_directories(include)

file(GLOB SRCFILES "*.cpp")

cupla_add_executable(gameOfLife
    ${SRCFILES}
)

target_link_libraries(gameOfLife PRIVATE ${LIBS})


################################################################################
# Install GoL
################################################################################

install(TARGETS gameOfLife
        RUNTIME DESTINATION bin)

set(GOL_COPY_ON_INSTALL "submit" CACHE LIST "folder which copied on install to install path" )

# copy all subfolders (defined in: GOL_COPY_ON_INSTALL) to install folder
foreach(dir ${GOL_COPY_ON_INSTALL})
  #if source not exists than copy
  if(NOT EXISTS "${CMAKE_INSTALL_PREFIX}/${dir}")
      # copy important subfolders from extension path (default picongpu parameter)
      if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dir}/")
        INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${dir}/."
          DESTINATION "${CMAKE_INSTALL_PREFIX}/${dir}"
          PATTERN .svn EXCLUDE
        )
      endif()
  endif()
endforeach()
