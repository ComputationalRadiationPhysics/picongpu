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


set(_TARGET_NAME "babelstream")
append_recursive_files_add_to_src_group("src/" "src/" "cpp" _FILES_SOURCE)

alpaka_add_executable(
    ${_TARGET_NAME}
    ${_FILES_SOURCE})

target_include_directories(
 ${_TARGET_NAME}
 PRIVATE "src")

target_link_libraries(
    ${_TARGET_NAME}
    PRIVATE common)

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER benchmarks/babelstream)

#Run as a ctest
if(alpaka_CI)
    # Only run for release builds since this is a benchmark
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
       add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME})
    endif()
else()
    # For a normal benchmark test, number of samples should be equal to the default value.
    add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME})
endif()
