#
# Copyright 2023 Erik Zenker, Benjamin Worpitz, Jan Stephan, Bernhard Manfred Gruber
# SPDX-License-Identifier: ISC
#

cmake_minimum_required(VERSION 3.22)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

project(babelstream LANGUAGES CXX)

if(NOT TARGET alpaka::alpaka)
    option(alpaka_USE_SOURCE_TREE "Use alpaka's source tree instead of an alpaka installation" OFF)
    if(alpaka_USE_SOURCE_TREE)
        # Don't build the benchmarks recursively
        set(alpaka_BUILD_BENCHMARKS OFF)
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../.." "${CMAKE_BINARY_DIR}/alpaka")
    else()
        find_package(alpaka REQUIRED)
    endif()
endif()

alpaka_add_executable(${PROJECT_NAME} src/main.cpp src/Stream.h src/AlpakaStream.cpp src/AlpakaStream.h)
target_compile_definitions(${PROJECT_NAME} PUBLIC ALPAKA)
target_link_libraries(${PROJECT_NAME} PUBLIC alpaka::alpaka)
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER benchmarks)

# add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME})
