#
# Copyright 2015-2020 Benjamin Worpitz, Jan Stephan
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

################################################################################
# Required CMake version

cmake_minimum_required(VERSION 3.15)

cmake_policy(SET CMP0091 OLD)

include(CMakePrintHelpers)

#-------------------------------------------------------------------------------
# Find alpaka version.
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/include/alpaka/version.hpp" ALPAKA_VERSION_MAJOR_HPP REGEX "#define ALPAKA_VERSION_MAJOR ")
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/include/alpaka/version.hpp" ALPAKA_VERSION_MINOR_HPP REGEX "#define ALPAKA_VERSION_MINOR ")
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/include/alpaka/version.hpp" ALPAKA_VERSION_PATCH_HPP REGEX "#define ALPAKA_VERSION_PATCH ")

string(REGEX MATCH "([0-9]+)" ALPAKA_VERSION_MAJOR  ${ALPAKA_VERSION_MAJOR_HPP})
string(REGEX MATCH "([0-9]+)" ALPAKA_VERSION_MINOR  ${ALPAKA_VERSION_MINOR_HPP})
string(REGEX MATCH "([0-9]+)" ALPAKA_VERSION_PATCH  ${ALPAKA_VERSION_PATCH_HPP})

set(PACKAGE_VERSION "${ALPAKA_VERSION_MAJOR}.${ALPAKA_VERSION_MINOR}.${ALPAKA_VERSION_PATCH}")

project(alpaka VERSION      ${ALPAKA_VERSION_MAJOR}.${ALPAKA_VERSION_MINOR}.${ALPAKA_VERSION_PATCH}
               DESCRIPTION  "The alpaka library is a header-only C++14 abstraction library for accelerator development."
               HOMEPAGE_URL "https://github.com/alpaka-group/alpaka"
               LANGUAGES    CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Options and Variants

option(alpaka_BUILD_EXAMPLES "Build the examples" OFF)

option(BUILD_TESTING "Build the testing tree." OFF)
include(CTest)

################################################################################
# Internal variables.

# Set found to true initially and set it to false if a required dependency is missing.
set(_ALPAKA_FOUND TRUE)

# This file's directory.
set(_ALPAKA_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})
# Normalize the path (e.g. remove ../)
get_filename_component(_ALPAKA_ROOT_DIR ${_ALPAKA_ROOT_DIR} ABSOLUTE)

# Add common functions.
set(_ALPAKA_COMMON_FILE "${_ALPAKA_ROOT_DIR}/cmake/common.cmake")
include(${_ALPAKA_COMMON_FILE})

# Add ALPAKA_ADD_EXECUTABLE function.
set(_ALPAKA_ADD_EXECUTABLE_FILE "${_ALPAKA_ROOT_DIR}/cmake/addExecutable.cmake")
include(${_ALPAKA_ADD_EXECUTABLE_FILE})

# Add ALPAKA_ADD_LIBRARY function.
set(_ALPAKA_ADD_LIBRARY_FILE "${_ALPAKA_ROOT_DIR}/cmake/addLibrary.cmake")
include(${_ALPAKA_ADD_LIBRARY_FILE})

# Add module search path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${_ALPAKA_ROOT_DIR}/cmake/modules/")

# parse environment variable `CMAKE_MODULE_PATH`
if(DEFINED ENV{CMAKE_MODULE_PATH})
    string(REPLACE ":" ";" ENV_CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})
    foreach(_PREFIX_PATH ${ENV_CMAKE_MODULE_PATH})
        set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${_PREFIX_PATH}")
    endforeach()
endif()

# Set include directories
set(_ALPAKA_INCLUDE_DIRECTORY "${_ALPAKA_ROOT_DIR}/include")
set(_ALPAKA_SUFFIXED_INCLUDE_DIR "${_ALPAKA_INCLUDE_DIRECTORY}/alpaka")

include(${_ALPAKA_ROOT_DIR}/cmake/alpakaCommon.cmake)

# Add all the source and include files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_src_group("${_ALPAKA_SUFFIXED_INCLUDE_DIR}" "${_ALPAKA_SUFFIXED_INCLUDE_DIR}" "hpp" _ALPAKA_FILES_HEADER)
append_recursive_files_add_to_src_group("${_ALPAKA_SUFFIXED_INCLUDE_DIR}" "${_ALPAKA_SUFFIXED_INCLUDE_DIR}" "h" _ALPAKA_FILES_HEADER)

append_recursive_files_add_to_src_group("${_ALPAKA_ROOT_DIR}/script" "${_ALPAKA_ROOT_DIR}" "sh" _ALPAKA_FILES_SCRIPT)
set_source_files_properties(${_ALPAKA_FILES_SCRIPT} PROPERTIES HEADER_FILE_ONLY TRUE)

append_recursive_files_add_to_src_group("${_ALPAKA_ROOT_DIR}/cmake" "${_ALPAKA_ROOT_DIR}" "cmake" _ALPAKA_FILES_CMAKE)
list(APPEND _ALPAKA_FILES_CMAKE "${_ALPAKA_ROOT_DIR}/cmake/alpakaConfig.cmake.in" "${_ALPAKA_ROOT_DIR}/CMakeLists.txt")
set_source_files_properties(${_ALPAKA_FILES_CMAKE} PROPERTIES HEADER_FILE_ONLY TRUE)

append_recursive_files_add_to_src_group("${_ALPAKA_ROOT_DIR}/docs/markdown" "${_ALPAKA_ROOT_DIR}" "md" _ALPAKA_FILES_DOC)
set_source_files_properties(${_ALPAKA_FILES_DOC} PROPERTIES HEADER_FILE_ONLY TRUE)

append_recursive_files_add_to_src_group("${_ALPAKA_ROOT_DIR}/.github" "${_ALPAKA_ROOT_DIR}" "yml" _ALPAKA_FILES_OTHER)
list(APPEND _ALPAKA_FILES_OTHER "${_ALPAKA_ROOT_DIR}/.clang-format" "${_ALPAKA_ROOT_DIR}/.gitignore" "${_ALPAKA_ROOT_DIR}/.zenodo.json" "${_ALPAKA_ROOT_DIR}/LICENSE" "${_ALPAKA_ROOT_DIR}/README.md")
set_source_files_properties(${_ALPAKA_FILES_OTHER} PROPERTIES HEADER_FILE_ONLY TRUE)

if(TARGET alpaka)
    # HACK: Workaround for the limitation that files added to INTERFACE targets (target_sources) can not be marked as PUBLIC or PRIVATE but only as INTERFACE.
    # Therefore those files will be added to projects "linking" to the INTERFACE library, but are not added to the project itself within an IDE.
    add_custom_target("alpakaIde"
                      SOURCES ${_ALPAKA_FILES_HEADER} ${_ALPAKA_FILES_SCRIPT} ${_ALPAKA_FILES_CMAKE} ${_ALPAKA_FILES_DOC} ${_ALPAKA_FILES_OTHER})
endif()

################################################################################
# Export NVCC/HIPCC flags to parent scope if alpaka is used as a CMake
# subdirectory.
#
# These flags are set in cmake/alpakaCommon.cmake but are visible in this scope
# since alpakaCommon.cmake is included.

if(NOT ${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME})
    if(ALPAKA_ACC_GPU_CUDA_ENABLE)
        # export NVCC flags to parent scope in case alpaka is another project's subdirectory
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} PARENT_SCOPE)
        set(CUDA_HOST_COMPILER ${CUDA_HOST_COMPILER} PARENT_SCOPE)
        set(CUDA_VERBOSE_BUILD ${CUDA_VERBOSE_BUILD} PARENT_SCOPE)
        set(CUDA_SEPARABLE_COMPILATION ${CUDA_SEPARABLE_COMPILATION} PARENT_SCOPE)
    endif()

    if(ALPAKA_ACC_GPU_HIP_ENABLE)
        # export HIPCC flags to parent scope in case alpaka is another project's subdirectory
        set(HIP_HIPCC_FLAGS ${HIP_HIPCC_FLAGS} PARENT_SCOPE)
        set(HIP_NVCC_FLAGS ${HIP_NVCC_FLAGS} PARENT_SCOPE)
        set(HIP_VERBOSE_BUILD ${HIP_VERBOSE_BUILD} PARENT_SCOPE)
    endif()

    if((ALPAKA_ACC_GPU_CUDA_ENABLE OR ALPAKA_ACC_GPU_HIP_ENABLE) AND ALPAKA_CUDA_COMPILER MATCHES "nvcc")
        set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} PARENT_SCOPE)
    endif()
endif()

################################################################################
# Add subdirectories

if(alpaka_BUILD_EXAMPLES)
    add_subdirectory("example/")
endif()
if(BUILD_TESTING)
    add_subdirectory("test/")
endif()

################################################################################
# Installation.

# Do not install if alpaka is used as a CMake subdirectory
if(${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME})
    include(CMakePackageConfigHelpers)
    include(GNUInstallDirs)

    set(_ALPAKA_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/alpaka")

    install(TARGETS alpaka
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    write_basic_package_version_file(
        "alpakaConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion)

    configure_package_config_file(
        "${_ALPAKA_ROOT_DIR}/cmake/alpakaConfig.cmake.in"
        "${PROJECT_BINARY_DIR}/alpakaConfig.cmake" 
        INSTALL_DESTINATION "${_ALPAKA_INSTALL_CMAKEDIR}")

    install(FILES "${PROJECT_BINARY_DIR}/alpakaConfig.cmake"
                  "${PROJECT_BINARY_DIR}/alpakaConfigVersion.cmake"
            DESTINATION "${_ALPAKA_INSTALL_CMAKEDIR}")

    install(DIRECTORY "${_ALPAKA_SUFFIXED_INCLUDE_DIR}"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

    install(FILES "${_ALPAKA_ROOT_DIR}/cmake/addExecutable.cmake"
                  "${_ALPAKA_ROOT_DIR}/cmake/addLibrary.cmake"
                  "${_ALPAKA_ROOT_DIR}/cmake/alpakaCommon.cmake"
                  "${_ALPAKA_ROOT_DIR}/cmake/common.cmake"
            DESTINATION "${_ALPAKA_INSTALL_CMAKEDIR}")

    install(DIRECTORY "${_ALPAKA_ROOT_DIR}/cmake/modules"
            DESTINATION "${_ALPAKA_INSTALL_CMAKEDIR}")
endif()
