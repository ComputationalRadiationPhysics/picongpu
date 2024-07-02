#
# Copyright 2023 Benjamin Worpitz, Jan Stephan, Mehmet Yusufoglu
# SPDX-License-Identifier: ISC
#

################################################################################
# Required CMake version.
################################################################################

cmake_minimum_required(VERSION 3.22)

project("alpakaBenchmarks" LANGUAGES CXX)

################################################################################
# Add subdirectories.
################################################################################

add_subdirectory("babelstream/")
