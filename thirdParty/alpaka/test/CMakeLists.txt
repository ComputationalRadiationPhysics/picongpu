#
# Copyright 2015-2019 Benjamin Worpitz, Axel Huebl
#
# This file is part of Alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

CMAKE_MINIMUM_REQUIRED(VERSION 3.11.0)

# Search in <PackageName>_ROOT:
# https://cmake.org/cmake/help/v3.12/policy/CMP0074.html
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

SET(ALPAKA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../" CACHE STRING "The location of the alpaka library")
LIST(APPEND CMAKE_MODULE_PATH "${ALPAKA_ROOT}")
FIND_PACKAGE(alpaka REQUIRED)

ADD_SUBDIRECTORY("common/")

OPTION(ALPAKA_USE_INTERNAL_CATCH2 "Use internally shipped Catch2" ON)

IF(ALPAKA_USE_INTERNAL_CATCH2)
    message(STATUS "Catch2: Using INTERNAL version 2.5.0")
ELSE()
    find_package(Catch2 2.4.0 CONFIG REQUIRED)
    set_target_properties(Catch2::Catch2 PROPERTIES IMPORTED_GLOBAL TRUE)
    message(STATUS "Catch2: Found version ${Catch2_VERSION}")
ENDIF()

add_library(CatchMain CatchMain.cpp)
# target_compile_features(CatchMain PUBLIC cxx_std_11)  # min C++11
set_target_properties(CatchMain PROPERTIES
    FOLDER "test"
    CXX_STANDARD 11  # exactly C++11
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
target_compile_definitions(CatchMain PUBLIC "CATCH_CONFIG_FAST_COMPILE")
IF(MSVC)
    target_compile_definitions(CatchMain PUBLIC "CATCH_CONFIG_WINDOWS_CRTDBG")
    target_compile_options(CatchMain PUBLIC "/bigobj")
ENDIF()
IF(ALPAKA_USE_INTERNAL_CATCH2)
    target_include_directories(CatchMain SYSTEM PUBLIC
        ${alpakaAll_SOURCE_DIR}/thirdParty/catch2/include)
ELSE()
    target_include_directories(CatchMain SYSTEM PUBLIC
        $<TARGET_PROPERTY:Catch2::Catch2,INTERFACE_INCLUDE_DIRECTORIES>)
ENDIF()
SET_TARGET_PROPERTIES(
    CatchMain
    PROPERTIES FOLDER "test")

# NVCC does not incorporate the COMPILE_OPTIONS of a target but only the CMAKE_CXX_FLAGS
IF(ALPAKA_ACC_GPU_CUDA_ENABLE AND NOT ALPAKA_CUDA_COMPILER MATCHES "clang")
    GET_TARGET_PROPERTY(_COMMON_COMPILE_OPTIONS common COMPILE_OPTIONS)
    # If the property does not exist, the variable is set to NOTFOUND.
    IF(_COMMON_COMPILE_OPTIONS)
        STRING(REPLACE ";" " " _COMMON_COMPILE_OPTIONS_STRING "${_COMMON_COMPILE_OPTIONS}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_COMMON_COMPILE_OPTIONS_STRING}")
    ENDIF()
ENDIF()

LIST(APPEND _ALPAKA_TEST_OPTIONS "--use-colour yes")

ADD_SUBDIRECTORY("analysis/")
ADD_SUBDIRECTORY("integ/")
ADD_SUBDIRECTORY("unit/")
