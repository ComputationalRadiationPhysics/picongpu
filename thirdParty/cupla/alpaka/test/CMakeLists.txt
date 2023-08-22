#
# Copyright 2023 Benjamin Worpitz, Axel Hübl, Jan Stephan
# SPDX-License-Identifier: MPL-2.0
#

cmake_minimum_required(VERSION 3.22)

if(NOT TARGET alpaka::alpaka)
    list(APPEND CMAKE_MODULE_PATH "${alpaka_ROOT}")
    find_package(alpaka REQUIRED)
endif()

add_subdirectory(common)

list(APPEND _alpaka_TEST_OPTIONS --colour-mode default)

add_subdirectory(analysis)
add_subdirectory(integ)
add_subdirectory(unit)
