#
# Copyright 2021 Benjamin Worpitz, Jan Stephan
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

################################################################################
# Required CMake version.
################################################################################

cmake_minimum_required(VERSION 3.18)

project("alpakaIntegTest" LANGUAGES CXX)

################################################################################
# Add subdirectories.
################################################################################

add_subdirectory("axpy/")
add_subdirectory("cudaOnly/")
add_subdirectory("hostOnlyAPI/")
add_subdirectory("mandelbrot/")
add_subdirectory("matMul/")
add_subdirectory("separableCompilation/")
add_subdirectory("sharedMem/")
add_subdirectory("zeroDimBuffer/")
