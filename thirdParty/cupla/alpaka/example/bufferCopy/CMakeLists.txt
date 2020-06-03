#
# Copyright 2014-2019 Erik Zenker, Benjamin Worpitz
#
# This file exemplifies usage of Alpaka.
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED “AS IS” AND ISC DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL ISC BE LIABLE FOR ANY
# SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
# IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#

################################################################################
# Required CMake version.

CMAKE_MINIMUM_REQUIRED(VERSION 3.11.4)

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Project.

SET(_TARGET_NAME bufferCopy)

PROJECT(${_TARGET_NAME})

################################################################################
# CMake policies
#
# Search in <PackageName>_ROOT:
#   https://cmake.org/cmake/help/v3.12/policy/CMP0074.html

if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

#-------------------------------------------------------------------------------
# Find alpaka.

SET(ALPAKA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../" CACHE STRING "The location of the alpaka library")
LIST(APPEND CMAKE_MODULE_PATH "${ALPAKA_ROOT}")
FIND_PACKAGE(alpaka REQUIRED)

#-------------------------------------------------------------------------------
# Add executable.

ALPAKA_ADD_EXECUTABLE(
    ${_TARGET_NAME}
    src/bufferCopy.cpp)
TARGET_LINK_LIBRARIES(
    ${_TARGET_NAME}
    PUBLIC alpaka)

SET_TARGET_PROPERTIES(${_TARGET_NAME} PROPERTIES FOLDER example)

ADD_TEST(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME})
