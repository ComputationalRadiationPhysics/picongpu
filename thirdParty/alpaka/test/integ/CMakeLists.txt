#
# Copyright 2023 Benjamin Worpitz, Jan Stephan
# SPDX-License-Identifier: MPL-2.0
#

################################################################################
# Required CMake version.
################################################################################

cmake_minimum_required(VERSION 3.22)

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
