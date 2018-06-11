#
# Copyright 2015-2017 Benjamin Worpitz
#
# This file is part of alpaka.
#
# alpaka is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# alpaka is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with alpaka.
# If not, see <http://www.gnu.org/licenses/>.
#

################################################################################
# Required CMake version

cmake_minimum_required(VERSION 3.7.0)

project("alpakaAll")

################################################################################
# Options and Variants

option(alpaka_BUILD_EXAMPLES "Build the examples" ON)

include(CTest)
# automatically defines: BUILD_TESTING, default is ON

################################################################################
# Add subdirectories.

if(alpaka_BUILD_EXAMPLES)
    add_subdirectory("example/")
endif()
if(BUILD_TESTING)
    add_subdirectory("test/")
endif()
