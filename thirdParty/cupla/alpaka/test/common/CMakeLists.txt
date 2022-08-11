#
# Copyright 2014-2020 Benjamin Worpitz, Jan Stephan
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

cmake_minimum_required(VERSION 3.18)

set(_COMMON_TARGET_NAME "common")
set(_COMMON_COMPILE_OPTIONS_FILE "devCompileOptions.cmake")

add_library(${_COMMON_TARGET_NAME} INTERFACE)

include(${_COMMON_COMPILE_OPTIONS_FILE})
message(STATUS "alpaka_DEV_COMPILE_OPTIONS: ${alpaka_DEV_COMPILE_OPTIONS}")
target_compile_options(${_COMMON_TARGET_NAME} INTERFACE ${alpaka_DEV_COMPILE_OPTIONS})

if(MSVC)
    target_compile_options(${_COMMON_TARGET_NAME} INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/wd4996> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4996>) # This function or variable may be unsafe. Consider using <safe_version> instead.
    target_compile_options(${_COMMON_TARGET_NAME} INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/bigobj> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
endif()

if(alpaka_ACC_GPU_CUDA_ENABLE OR (alpaka_ACC_GPU_HIP_ENABLE AND HIP_PLATFORM MATCHES "nvcc"))
    # CUDA driver API is used by EventHostManualTrigger
    target_link_libraries(${_COMMON_TARGET_NAME} INTERFACE CUDA::cuda_driver)
    target_compile_definitions(${_COMMON_TARGET_NAME} INTERFACE "CUDA_API_PER_THREAD_DEFAULT_STREAM")
endif()
if(alpaka_ACC_GPU_CUDA_ENABLE AND CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
    # nvcc supports werror starting with 10.2
    if(CUDA_VERSION GREATER_EQUAL 10.2)
        message(STATUS "adding -Werror=all-warnings")
        target_compile_options(${_COMMON_TARGET_NAME} INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Werror=all-warnings>)
    endif()
endif()

target_link_libraries(${_COMMON_TARGET_NAME} INTERFACE alpaka::alpaka)

# Prevent warnings from shady code inside Catch2
get_target_property(alpaka_CATCH2_INCLUDE_DIRS Catch2::Catch2 INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(${_COMMON_TARGET_NAME} SYSTEM INTERFACE ${alpaka_CATCH2_INCLUDE_DIRS})

# Now we can safely link to Catch2
target_link_libraries(${_COMMON_TARGET_NAME} INTERFACE Catch2::Catch2WithMain)

if(TARGET ${_COMMON_TARGET_NAME})
    # HACK: Workaround for the limitation that files added to INTERFACE targets (target_sources) can not be marked as PUBLIC or PRIVATE but only as INTERFACE.
    # Therefore those files will be added to projects "linking" to the INTERFACE library, but are not added to the project itself within an IDE.
    add_custom_target("${_COMMON_TARGET_NAME}Ide" SOURCES ${_COMMON_FILES_HEADER} ${_COMMON_COMPILE_OPTIONS_FILE})
    set_target_properties("${_COMMON_TARGET_NAME}Ide" PROPERTIES FOLDER "test")
endif()
