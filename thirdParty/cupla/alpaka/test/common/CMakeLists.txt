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
set(_COMMON_COMPILE_OPTIONS_FILE "devCompileOptions.cmake")
set(_COMMON_INCLUDE_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/include")

# Add all the source files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_src_group("${_COMMON_INCLUDE_DIRECTORY}" "${_COMMON_INCLUDE_DIRECTORY}" "hpp" _COMMON_FILES_HEADER)

add_library(${_COMMON_TARGET_NAME} INTERFACE)

target_include_directories(${_COMMON_TARGET_NAME} INTERFACE ${_COMMON_INCLUDE_DIRECTORY})

include(${_COMMON_COMPILE_OPTIONS_FILE})
target_compile_options(${_COMMON_TARGET_NAME} INTERFACE ${ALPAKA_DEV_COMPILE_OPTIONS})

if(MSVC)
    target_compile_options(${_COMMON_TARGET_NAME} INTERFACE "/wd4996") # This function or variable may be unsafe. Consider using <safe_version> instead.
    target_compile_options(${_COMMON_TARGET_NAME} INTERFACE "/bigobj")
endif()

if(ALPAKA_ACC_GPU_CUDA_ENABLE OR (ALPAKA_ACC_GPU_HIP_ENABLE AND HIP_PLATFORM MATCHES "nvcc"))
    # CUDA driver API is used by EventHostManualTrigger
    target_link_libraries(${_COMMON_TARGET_NAME} INTERFACE "${CUDA_CUDA_LIBRARY}")
    target_compile_definitions(${_COMMON_TARGET_NAME} INTERFACE "CUDA_API_PER_THREAD_DEFAULT_STREAM")
endif()

target_link_libraries(${_COMMON_TARGET_NAME} INTERFACE alpaka::alpaka)
target_link_libraries(${_COMMON_TARGET_NAME} INTERFACE CatchMain)

if(TARGET ${_COMMON_TARGET_NAME})
    # HACK: Workaround for the limitation that files added to INTERFACE targets (target_sources) can not be marked as PUBLIC or PRIVATE but only as INTERFACE.
    # Therefore those files will be added to projects "linking" to the INTERFACE library, but are not added to the project itself within an IDE.
    add_custom_target("${_COMMON_TARGET_NAME}Ide" SOURCES ${_COMMON_FILES_HEADER} ${_COMMON_COMPILE_OPTIONS_FILE})
    set_target_properties("${_COMMON_TARGET_NAME}Ide" PROPERTIES FOLDER "test")
endif()
