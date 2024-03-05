# Copyright 2023 Bernhard Manfred Gruber
# SPDX-License-Identifier: ISC

cmake_minimum_required(VERSION 3.22)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

project(alpaka-ls LANGUAGES CXX)
if(NOT TARGET alpaka::alpaka)
    option(alpaka_USE_SOURCE_TREE "Use alpaka's source tree instead of an alpaka installation" OFF)
    if(alpaka_USE_SOURCE_TREE)
        set(alpaka_BUILD_EXAMPLES OFF) # Don't build the examples recursively
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../.." "${CMAKE_BINARY_DIR}/alpaka")
    else()
        find_package(alpaka REQUIRED)
    endif()
endif()

alpaka_add_executable(${PROJECT_NAME} src/ls.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC alpaka::alpaka)
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER example)
add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
