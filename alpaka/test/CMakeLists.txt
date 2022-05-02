#
# Copyright 2015-2020 Benjamin Worpitz, Axel Huebl, Jan Stephan
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

cmake_minimum_required(VERSION 3.18)

add_subdirectory(catch_main)

if(NOT TARGET alpaka::alpaka)
    list(APPEND CMAKE_MODULE_PATH "${alpaka_ROOT}")
    find_package(alpaka REQUIRED)
endif()

add_subdirectory(common)

list(APPEND _alpaka_TEST_OPTIONS --use-colour yes)

add_subdirectory(analysis)
add_subdirectory(integ)
add_subdirectory(unit)
