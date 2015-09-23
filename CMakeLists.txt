# Copyright 2013-2015 Felix Schmitt, Heiko Burau, Rene Widera, Axel Huebl
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

cmake_minimum_required(VERSION 2.8.12.2)


################################################################################
# Project
################################################################################

project(PIConGPU2_full_build)


################################################################################
# Disallow in-source build
################################################################################

string(FIND "${PIConGPU2_full_build_BINARY_DIR}"
            "${PIConGPU2_full_build_SOURCE_DIR}" IN_SRC_POS)
if(IN_SRC_POS GREATER -1)
  message(FATAL_ERROR
    "PICoNGPU requires an out of source build. "
    "Please remove \n"
    "  - CMakeCache.txt\n"
    "  - CMakeFiles/\n"
    "and create a separate build directory. "
    "See: doc/INSTALL.md")
endif()

unset(IN_SRC_POS)


################################################################################
# Add PICoNGPU project from sub directory
################################################################################

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "install prefix" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

add_subdirectory("${CMAKE_SOURCE_DIR}/src/picongpu" "${CMAKE_BINARY_DIR}/build_picongpu")
