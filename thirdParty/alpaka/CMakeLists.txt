#
# Copyright 2015-2019 Benjamin Worpitz
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

################################################################################
# Required CMake version

cmake_minimum_required(VERSION 3.11.4)

project("alpakaAll")

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# CMake policies
#
# Search in <PackageName>_ROOT:
#   https://cmake.org/cmake/help/v3.12/policy/CMP0074.html

if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

################################################################################
# Options and Variants

option(alpaka_BUILD_EXAMPLES "Build the examples" ON)

include(CTest)
# automatically defines: BUILD_TESTING, default is ON

################################################################################
# Add subdirectories

if(alpaka_BUILD_EXAMPLES)
    add_subdirectory("example/")
endif()
if(BUILD_TESTING)
    add_subdirectory("test/")
endif()
