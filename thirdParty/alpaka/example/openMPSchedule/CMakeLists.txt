#
# Copyright 2023 Erik Zenker, Benjamin Worpitz, Jan Stephan, Sergei Bastrakov
# SPDX-License-Identifier: ISC
#

################################################################################
# Required CMake version.

cmake_minimum_required(VERSION 3.22)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Project.

set(_TARGET_NAME openMPSchedule)

project(${_TARGET_NAME} LANGUAGES CXX)

#-------------------------------------------------------------------------------
# Find alpaka.

if(NOT TARGET alpaka::alpaka)
    option(alpaka_USE_SOURCE_TREE "Use alpaka's source tree instead of an alpaka installation" OFF)

    if(alpaka_USE_SOURCE_TREE)
        # Don't build the examples recursively
        set(alpaka_BUILD_EXAMPLES OFF)
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../.." "${CMAKE_BINARY_DIR}/alpaka")
    else()
        find_package(alpaka REQUIRED)
    endif()
endif()

#-------------------------------------------------------------------------------
# Add executable.

alpaka_add_executable(
    ${_TARGET_NAME}
    src/openMPSchedule.cpp)
target_link_libraries(
    ${_TARGET_NAME}
    PUBLIC alpaka::alpaka)

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER example)

add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME})
