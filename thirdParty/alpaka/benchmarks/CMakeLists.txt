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

if(NOT BUILD_TESTING)
    # Testing is not enabled therefore CATCH2 which is part of common must be pulled.
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../test/common "${CMAKE_BINARY_DIR}/test/common")
endif()

add_subdirectory("babelstream/")
