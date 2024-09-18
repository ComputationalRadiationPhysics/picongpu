#
# Copyright 2014-2023 Benjamin Worpitz, Axel Huebl, Jan Stephan, Rene Widera
#
# This file is part of PIConGPU.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

cmake_minimum_required(VERSION 3.22)
project(PIConGPUHeaderTest)

# Add common functions from alpaka.
include(${CMAKE_CURRENT_SOURCE_DIR}/../../thirdParty/alpaka/cmake/common.cmake)

################################################################################
# Directory of this file.
################################################################################
set(PIC_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../include/picongpu)

# Normalize the path (e.g. remove ../)
get_filename_component(PIC_ROOT_DIR "${PIC_ROOT_DIR}" ABSOLUTE)

###############################################################################
# Language Flags
###############################################################################

# enforce C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)

################################################################################
# PMacc
################################################################################

find_package(PMacc REQUIRED CONFIG PATHS "${PIC_ROOT_DIR}/../pmacc")

################################################################################
# JSON
################################################################################

# none of these should appear in cmake-gui, so make them internal
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_MultipleHeaders OFF CACHE INTERNAL "")
set(JSON_ImplicitConversions OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")  # only used PRIVATE

# allow using externally installed nlohmann_json
set(
        PIC_nlohmann_json_PROVIDER "intern" CACHE
        STRING "Use internally shipped or external nlohmann_json library.")
set_property(
        CACHE PIC_nlohmann_json_PROVIDER
        PROPERTY STRINGS "intern;extern")
mark_as_advanced(PIC_nlohmann_json_PROVIDER)
if(${PIC_nlohmann_json_PROVIDER} STREQUAL "intern")
    add_subdirectory(
            "${PIC_ROOT_DIR}/../../thirdParty/nlohmann_json"
            "${CMAKE_CURRENT_BINARY_DIR}/build_nlohmann_json")
else()
    find_package(nlohmann_json 3.9.1 CONFIG REQUIRED)
    message(STATUS "nlohmann-json: Found version '${nlohmann_json_VERSION}'")
endif()


set(_TARGET_NAME "PIConGPU-HeaderCheckTest")

###############################################################################
# Catch2
###############################################################################

add_subdirectory(${PIC_ROOT_DIR}/../../thirdParty/catch2 ${CMAKE_BINARY_DIR}/catch2)

#---------------------------------------------------------------------------
# Create source files.
set(PIC_SUFFIXED_INCLUDE_DIR "${PIC_ROOT_DIR}")
append_recursive_files("${PIC_SUFFIXED_INCLUDE_DIR}/algorithms" "hpp" "PIC_FILES_HEADER")
append_recursive_files("${PIC_SUFFIXED_INCLUDE_DIR}/debug" "hpp" "PIC_FILES_HEADER")
append_recursive_files("${PIC_SUFFIXED_INCLUDE_DIR}/random" "hpp" "PIC_FILES_HEADER")
append_recursive_files("${PIC_SUFFIXED_INCLUDE_DIR}/traits" "hpp" "PIC_FILES_HEADER")

# remove traits which where include dependencies can not be solved easily
file(GLOB_RECURSE PIC_ATTRIBUTE_TRAITS "${PIC_SUFFIXED_INCLUDE_DIR}/traits/attribute/*.hpp")
file(GLOB_RECURSE PIC_FRAME_TRAITS "${PIC_SUFFIXED_INCLUDE_DIR}/traits/frame/*.hpp")
list(REMOVE_ITEM PIC_FILES_HEADER ${PIC_ATTRIBUTE_TRAITS})
list(REMOVE_ITEM PIC_FILES_HEADER ${PIC_FRAME_TRAITS})

set(_GENERATED_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/src")

file(REMOVE_RECURSE ${_GENERATED_SOURCE_DIR})

foreach (_HEADER_FILE ${PIC_FILES_HEADER})
    # Remove the parent directory from the path.
    # NOTE: This is not correct because it does not only replace at the beginning of the string.
    #  "STRING(REGEX REPLACE" would be correct if there was an easy way to escape arbitrary strings.
    string(
            REPLACE "${PIC_SUFFIXED_INCLUDE_DIR}/" ""
            _HEADER_FILE
            "${_HEADER_FILE}")
    set(_SOURCE_FILE "${_GENERATED_SOURCE_DIR}/${_HEADER_FILE}.cpp")
    file(WRITE "${_SOURCE_FILE}" "#include <picongpu/${_HEADER_FILE}>")
endforeach ()

#---------------------------------------------------------------------------
# Add executable.

append_recursive_files_add_to_src_group("${_GENERATED_SOURCE_DIR}" "${_GENERATED_SOURCE_DIR}" "cpp" "_FILES_SOURCE")
list(APPEND _FILES_SOURCE "src/main.cpp")

# Always add all files to the target executable build call to add them to the build project.
alpaka_add_executable(
        ${_TARGET_NAME}
        ${_FILES_SOURCE})

target_link_libraries(${_TARGET_NAME} PUBLIC Catch2 Catch2WithMain)
target_link_libraries(${_TARGET_NAME} PRIVATE pmacc::pmacc)
target_include_directories(
        ${_TARGET_NAME}
        SYSTEM PRIVATE
        $<TARGET_PROPERTY:nlohmann_json::nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(${_TARGET_NAME} PRIVATE nlohmann_json::nlohmann_json)

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER "picongpu-headerCheck")

add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME})
