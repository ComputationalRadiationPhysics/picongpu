#
# Copyright 2014-2020 Benjamin Worpitz, Jan Stephan
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

cmake_minimum_required(VERSION 3.15)

set(_COMMON_TARGET_NAME "common")
set(_COMMON_INCLUDE_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/include")
set(_COMMON_SOURCE_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/src")

# Add all the source files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_src_group("${_COMMON_INCLUDE_DIRECTORY}" "${_COMMON_INCLUDE_DIRECTORY}" "hpp" _COMMON_FILES_HEADER)
append_recursive_files_add_to_src_group("${_COMMON_SOURCE_DIRECTORY}" "${_COMMON_SOURCE_DIRECTORY}" "cpp" _COMMON_FILES_SOURCE)

add_library(${_COMMON_TARGET_NAME} STATIC
            ${_COMMON_FILES_HEADER}
            ${_COMMON_FILES_SOURCE}
            "devCompileOptions.cmake")

target_link_libraries(${_COMMON_TARGET_NAME}
                      PUBLIC alpaka::alpaka)

target_include_directories(${_COMMON_TARGET_NAME}
                           PUBLIC ${_COMMON_INCLUDE_DIRECTORY})

include("devCompileOptions.cmake")
target_compile_options(${_COMMON_TARGET_NAME}
                       PUBLIC ${ALPAKA_DEV_COMPILE_OPTIONS})

if(MSVC)
    target_compile_options(${_COMMON_TARGET_NAME} PUBLIC "/wd4996") # This function or variable may be unsafe. Consider using <safe_version> instead.
    target_compile_options(${_COMMON_TARGET_NAME} PUBLIC "/bigobj")
endif()

if(ALPAKA_ACC_GPU_CUDA_ENABLE OR (ALPAKA_ACC_GPU_HIP_ENABLE AND HIP_PLATFORM MATCHES "nvcc"))
    # CUDA driver API is used by EventHostManualTrigger
    target_link_libraries(${_COMMON_TARGET_NAME} PUBLIC "${CUDA_CUDA_LIBRARY}")
    target_compile_definitions(${_COMMON_TARGET_NAME} PUBLIC "CUDA_API_PER_THREAD_DEFAULT_STREAM")
endif()

set_target_properties(${_COMMON_TARGET_NAME} PROPERTIES FOLDER "test")

target_link_libraries(${_COMMON_TARGET_NAME}
                      PUBLIC CatchMain)
