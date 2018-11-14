#
# Copyright 2014-2018 Benjamin Worpitz
#
# This file is part of alpaka.
#
# alpaka is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# alpaka is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with alpaka.
# If not, see <http://www.gnu.org/licenses/>.
#

CMAKE_MINIMUM_REQUIRED(VERSION 3.7.0)

SET(_COMMON_TARGET_NAME "common")

SET(_COMMON_INCLUDE_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/include")
LIST(APPEND _COMMON_INCLUDE_DIRECTORIES_PUBLIC "${_COMMON_INCLUDE_DIRECTORY}")
SET(_COMMON_SOURCE_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/src")

# Add all the source files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_src_group("${_COMMON_INCLUDE_DIRECTORY}" "${_COMMON_INCLUDE_DIRECTORY}" "hpp" _COMMON_FILES_HEADER)
append_recursive_files_add_to_src_group("${_COMMON_SOURCE_DIRECTORY}" "${_COMMON_SOURCE_DIRECTORY}" "cpp" _COMMON_FILES_SOURCE)

INCLUDE("${ALPAKA_ROOT}cmake/dev.cmake")
LIST(APPEND _COMMON_COMPILE_OPTIONS_PUBLIC ${ALPAKA_DEV_COMPILE_OPTIONS})
IF(MSVC)
    LIST(APPEND _COMMON_COMPILE_OPTIONS_PUBLIC "/bigobj")
    LIST(APPEND _COMMON_COMPILE_OPTIONS_PUBLIC "/wd4996")   # This function or variable may be unsafe. Consider using <safe_version> instead.
ENDIF()

IF(ALPAKA_ACC_GPU_CUDA_ENABLE OR (ALPAKA_ACC_GPU_HIP_ENABLE AND HIP_PLATFORM MATCHES "nvcc"))
    # CUDA driver API is used by EventHostManualTrigger
    LIST(APPEND _COMMON_LINK_LIBRARIES_PUBLIC "general;${CUDA_CUDA_LIBRARY}")
    LIST(APPEND _COMMON_COMPILE_DEFINITIONS_PUBLIC "CUDA_API_PER_THREAD_DEFAULT_STREAM")
ENDIF()

ADD_LIBRARY(
    ${_COMMON_TARGET_NAME}
    STATIC
    ${_COMMON_FILES_HEADER} ${_COMMON_FILES_SOURCE})
TARGET_INCLUDE_DIRECTORIES(
    ${_COMMON_TARGET_NAME}
    PUBLIC ${_COMMON_INCLUDE_DIRECTORIES_PUBLIC})
LIST(
    LENGTH
    _COMMON_COMPILE_DEFINITIONS_PUBLIC
    _COMMON_COMPILE_DEFINITIONS_PUBLIC_LENGTH)
IF(${_COMMON_COMPILE_DEFINITIONS_PUBLIC_LENGTH} GREATER 0)
    TARGET_COMPILE_DEFINITIONS(
        ${_COMMON_TARGET_NAME}
        PUBLIC ${_COMMON_COMPILE_DEFINITIONS_PUBLIC})
ENDIF()
TARGET_COMPILE_OPTIONS(
    ${_COMMON_TARGET_NAME}
    PUBLIC ${_COMMON_COMPILE_OPTIONS_PUBLIC})
TARGET_LINK_LIBRARIES(
    ${_COMMON_TARGET_NAME}
    PUBLIC alpaka ${_COMMON_LINK_LIBRARIES_PUBLIC})
SET_TARGET_PROPERTIES(
    ${_COMMON_TARGET_NAME}
    PROPERTIES FOLDER "test")
