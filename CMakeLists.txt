#
# Copyright 2023 Benjamin Worpitz, Jan Stephan, Bernhard Manfred Gruber
# SPDX-License-Identifier: MPL-2.0
#

################################################################################
# Required CMake version

cmake_minimum_required(VERSION 3.22)

cmake_policy(SET CMP0091 NEW)

include(CMakePrintHelpers)

#-------------------------------------------------------------------------------
# Find alpaka version.
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/include/alpaka/version.hpp" alpaka_VERSION_MAJOR_HPP REGEX "#define ALPAKA_VERSION_MAJOR ")
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/include/alpaka/version.hpp" alpaka_VERSION_MINOR_HPP REGEX "#define ALPAKA_VERSION_MINOR ")
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/include/alpaka/version.hpp" alpaka_VERSION_PATCH_HPP REGEX "#define ALPAKA_VERSION_PATCH ")

string(REGEX MATCH "([0-9]+)" alpaka_VERSION_MAJOR  ${alpaka_VERSION_MAJOR_HPP})
string(REGEX MATCH "([0-9]+)" alpaka_VERSION_MINOR  ${alpaka_VERSION_MINOR_HPP})
string(REGEX MATCH "([0-9]+)" alpaka_VERSION_PATCH  ${alpaka_VERSION_PATCH_HPP})

set(PACKAGE_VERSION "${alpaka_VERSION_MAJOR}.${alpaka_VERSION_MINOR}.${alpaka_VERSION_PATCH}")

project(alpaka VERSION      ${alpaka_VERSION_MAJOR}.${alpaka_VERSION_MINOR}.${alpaka_VERSION_PATCH}
               DESCRIPTION  "The alpaka library is a header-only C++17 abstraction library for accelerator development."
               HOMEPAGE_URL "https://github.com/alpaka-group/alpaka"
               LANGUAGES    CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Options and Variants

option(alpaka_BUILD_EXAMPLES "Build the examples" OFF)
option(alpaka_BUILD_BENCHMARKS "Build the benchmarks" OFF)

# Enable the test infrastructure only if alpaka is the top-level project
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(alpaka_ENABLE_WERROR "Treat all warnings as errors." OFF)
    option(BUILD_TESTING "Build the testing tree." OFF)
    include(CTest)
endif()

option(alpaka_INSTALL_TEST_HEADER "Install headers of the namespace alpaka::test. Attention, headers are not designed for production code, see documentation." OFF)

include(CMakeDependentOption)
cmake_dependent_option(alpaka_CHECK_HEADERS "Check all alpaka headers as part of the tests whether they can be compiled standalone." OFF BUILD_TESTING OFF)
cmake_dependent_option(alpaka_USE_INTERNAL_CATCH2 "Use internally shipped Catch2" ON BUILD_TESTING OFF)

################################################################################
# Internal variables.

# Set found to true initially and set it to false if a required dependency is missing.
set(_alpaka_FOUND TRUE)

# This file's directory.
set(_alpaka_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})
# Normalize the path (e.g. remove ../)
get_filename_component(_alpaka_ROOT_DIR ${_alpaka_ROOT_DIR} ABSOLUTE)

# Compiler feature tests.
set(_alpaka_FEATURE_TESTS_DIR "${_alpaka_ROOT_DIR}/cmake/tests")

# Add common functions.
set(_alpaka_COMMON_FILE "${_alpaka_ROOT_DIR}/cmake/common.cmake")
include(${_alpaka_COMMON_FILE})

# Add alpaka_ADD_EXECUTABLE function.
set(_alpaka_ADD_EXECUTABLE_FILE "${_alpaka_ROOT_DIR}/cmake/addExecutable.cmake")
include(${_alpaka_ADD_EXECUTABLE_FILE})

# Add alpaka_ADD_LIBRARY function.
set(_alpaka_ADD_LIBRARY_FILE "${_alpaka_ROOT_DIR}/cmake/addLibrary.cmake")
include(${_alpaka_ADD_LIBRARY_FILE})

# Set include directories
set(_alpaka_INCLUDE_DIRECTORY "${_alpaka_ROOT_DIR}/include")
set(_alpaka_SUFFIXED_INCLUDE_DIR "${_alpaka_INCLUDE_DIRECTORY}/alpaka")

# the sequential accelerator is required for the tests and examples
if(alpaka_BUILD_EXAMPLES OR alpaka_BUILD_BENCHMARKS OR BUILD_TESTING)
  if (NOT (alpaka_ACC_GPU_CUDA_ONLY_MODE OR alpaka_ACC_GPU_HIP_ONLY_MODE))
    if (NOT DEFINED alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE)
      option(alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE "enable alpaka serial accelerator" ON)
    elseif(NOT alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE)
      set(alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE ON CACHE BOOL "enable alpaka serial accelerator" FORCE)
    endif()
  else() #CUDA or HIP-only mode
    message(WARNING "CUDA/HIP-only mode enabled: not enabling alpaka serial accelerator (required for examples and tests)")
  endif()
endif()
include(${_alpaka_ROOT_DIR}/cmake/alpakaCommon.cmake)

# Add all the source and include files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_src_group("${_alpaka_SUFFIXED_INCLUDE_DIR}" "${_alpaka_SUFFIXED_INCLUDE_DIR}" "hpp" _alpaka_FILES_HEADER)
append_recursive_files_add_to_src_group("${_alpaka_SUFFIXED_INCLUDE_DIR}" "${_alpaka_SUFFIXED_INCLUDE_DIR}" "h" _alpaka_FILES_HEADER)

# remove headers of the folder alpaka/test, if alpaka_INSTALL_TEST_HEADER is disabled
if(NOT alpaka_INSTALL_TEST_HEADER)
  list(FILTER _alpaka_FILES_HEADER EXCLUDE REGEX "(.*)/alpaka/test/(.*)")
endif()

append_recursive_files_add_to_src_group("${_alpaka_ROOT_DIR}/script" "${_alpaka_ROOT_DIR}" "sh" _alpaka_FILES_SCRIPT)
set_source_files_properties(${_alpaka_FILES_SCRIPT} PROPERTIES HEADER_FILE_ONLY TRUE)

append_recursive_files_add_to_src_group("${_alpaka_ROOT_DIR}/cmake" "${_alpaka_ROOT_DIR}" "cmake" _alpaka_FILES_CMAKE)
list(APPEND _alpaka_FILES_CMAKE "${_alpaka_ROOT_DIR}/cmake/alpakaConfig.cmake.in" "${_alpaka_ROOT_DIR}/CMakeLists.txt")
set_source_files_properties(${_alpaka_FILES_CMAKE} PROPERTIES HEADER_FILE_ONLY TRUE)

append_recursive_files_add_to_src_group("${_alpaka_ROOT_DIR}/docs/markdown" "${_alpaka_ROOT_DIR}" "md" _alpaka_FILES_DOC)
set_source_files_properties(${_alpaka_FILES_DOC} PROPERTIES HEADER_FILE_ONLY TRUE)

append_recursive_files_add_to_src_group("${_alpaka_ROOT_DIR}/.github" "${_alpaka_ROOT_DIR}" "yml" _alpaka_FILES_OTHER)
list(APPEND _alpaka_FILES_OTHER "${_alpaka_ROOT_DIR}/.clang-format" "${_alpaka_ROOT_DIR}/.gitignore" "${_alpaka_ROOT_DIR}/.zenodo.json" "${_alpaka_ROOT_DIR}/LICENSE" "${_alpaka_ROOT_DIR}/README.md")
set_source_files_properties(${_alpaka_FILES_OTHER} PROPERTIES HEADER_FILE_ONLY TRUE)

if(TARGET alpaka)
    # HACK: Workaround for the limitation that files added to INTERFACE targets (target_sources) can not be marked as PUBLIC or PRIVATE but only as INTERFACE.
    # Therefore those files will be added to projects "linking" to the INTERFACE library, but are not added to the project itself within an IDE.
    add_custom_target("alpakaIde"
                      SOURCES ${_alpaka_FILES_HEADER} ${_alpaka_FILES_SCRIPT} ${_alpaka_FILES_CMAKE} ${_alpaka_FILES_DOC} ${_alpaka_FILES_OTHER})
endif()

################################################################################
# Export NVCC/HIPCC flags to parent scope if alpaka is used as a CMake
# subdirectory.
#
# These flags are set in cmake/alpakaCommon.cmake but are visible in this scope
# since alpakaCommon.cmake is included.

if(NOT CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    if(alpaka_ACC_GPU_HIP_ENABLE)
        # export HIPCC flags to parent scope in case alpaka is another project's subdirectory
        set(HIP_HIPCC_FLAGS ${HIP_HIPCC_FLAGS} PARENT_SCOPE)
        set(HIP_NVCC_FLAGS ${HIP_NVCC_FLAGS} PARENT_SCOPE)
        set(HIP_VERBOSE_BUILD ${HIP_VERBOSE_BUILD} PARENT_SCOPE)
    endif()
endif()

################################################################################
# Add subdirectories

add_subdirectory(thirdParty)

if(alpaka_BUILD_EXAMPLES)
    add_subdirectory("example/")
endif()

if(alpaka_BUILD_BENCHMARKS)
    add_subdirectory("benchmarks/")
endif()

# Only build the tests if alpaka is the top-level project and BUILD_TESTING is ON
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    add_subdirectory("test/")
endif()

################################################################################
# Installation.

# Do not install if alpaka is used as a CMake subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CMakePackageConfigHelpers)
    include(GNUInstallDirs)

    set(_alpaka_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/alpaka")

    install(TARGETS alpaka
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    write_basic_package_version_file(
        "alpakaConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion)

    configure_package_config_file(
        "${_alpaka_ROOT_DIR}/cmake/alpakaConfig.cmake.in"
        "${PROJECT_BINARY_DIR}/alpakaConfig.cmake"
        INSTALL_DESTINATION "${_alpaka_INSTALL_CMAKEDIR}")

    install(FILES "${PROJECT_BINARY_DIR}/alpakaConfig.cmake"
                  "${PROJECT_BINARY_DIR}/alpakaConfigVersion.cmake"
            DESTINATION "${_alpaka_INSTALL_CMAKEDIR}")

    if(alpaka_INSTALL_TEST_HEADER)
        install(DIRECTORY "${_alpaka_SUFFIXED_INCLUDE_DIR}"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
    else()
    install(DIRECTORY "${_alpaka_SUFFIXED_INCLUDE_DIR}"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
                PATTERN "test" EXCLUDE)
    endif()

    install(FILES "${_alpaka_ROOT_DIR}/cmake/addExecutable.cmake"
                  "${_alpaka_ROOT_DIR}/cmake/addLibrary.cmake"
                  "${_alpaka_ROOT_DIR}/cmake/alpakaCommon.cmake"
                  "${_alpaka_ROOT_DIR}/cmake/common.cmake"
            DESTINATION "${_alpaka_INSTALL_CMAKEDIR}")
    install(DIRECTORY "${_alpaka_ROOT_DIR}/cmake/tests"
            DESTINATION "${_alpaka_INSTALL_CMAKEDIR}")
endif()
