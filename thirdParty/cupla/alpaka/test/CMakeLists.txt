#
# Copyright 2015-2020 Benjamin Worpitz, Axel Huebl, Jan Stephan
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

cmake_minimum_required(VERSION 3.15)

add_subdirectory(catch_main)

if(NOT TARGET alpaka::alpaka)
    list(APPEND CMAKE_MODULE_PATH "${ALPAKA_ROOT}")
    find_package(alpaka REQUIRED)
endif()

add_subdirectory(common)

if(ALPAKA_ACC_GPU_CUDA_ENABLE AND ALPAKA_CUDA_COMPILER MATCHES "nvcc")
    # NVCC does not incorporate the INTERFACE_COMPILE_OPTIONS of a target but only the CMAKE_CXX_FLAGS
    get_target_property(_COMMON_COMPILE_OPTIONS common INTERFACE_COMPILE_OPTIONS)
    # If the property does not exist, the variable is set to NOTFOUND.
    if(_COMMON_COMPILE_OPTIONS)
        string(REPLACE ";" " " _COMMON_COMPILE_OPTIONS_STRING "${_COMMON_COMPILE_OPTIONS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_COMMON_COMPILE_OPTIONS_STRING}")
    endif()
    # nvcc supports werror starting with 10.2
    if(CUDA_VERSION GREATER_EQUAL 10.2)
        message("adding -Werror=all-warnings")
        list(APPEND CUDA_NVCC_FLAGS -Werror=all-warnings)
    endif()
endif()

list(APPEND _ALPAKA_TEST_OPTIONS --use-colour yes)

add_subdirectory(analysis)
add_subdirectory(integ)
add_subdirectory(unit)
