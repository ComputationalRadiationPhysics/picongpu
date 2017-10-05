#
# Copyright 2014-2015 Benjamin Worpitz
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
# Required CMake version.
################################################################################

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

PROJECT("alpakaIntegTest")

################################################################################
# Add subdirectories.
################################################################################

ADD_SUBDIRECTORY("axpy/")
ADD_SUBDIRECTORY("cudaOnly/")
ADD_SUBDIRECTORY("mandelbrot/")
ADD_SUBDIRECTORY("matMul/")
ADD_SUBDIRECTORY("sharedMem/")
