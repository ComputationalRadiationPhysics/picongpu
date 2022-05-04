#
# Copyright 2014-2019 Benjamin Worpitz
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

CMAKE_MINIMUM_REQUIRED(VERSION 3.18)

#------------------------------------------------------------------------------
# Calls HIP_ADD_EXECUTABLE or ADD_EXECUTABLE depending on the enabled alpaka accelerators.
# Using a macro to stay in the scope (fixes lost assignment of linker command in FindHIP.cmake)
# https://github.com/ROCm-Developer-Tools/HIP/issues/631
MACRO(ALPAKA_ADD_EXECUTABLE In_Name)
    IF(alpaka_ACC_GPU_CUDA_ENABLE)
        ENABLE_LANGUAGE(CUDA)
        FOREACH(_file ${ARGN})
            IF((${_file} MATCHES "\\.cpp$") OR (${_file} MATCHES "\\.cxx$") OR (${_file} MATCHES "\\.cu$"))
                SET_SOURCE_FILES_PROPERTIES(${_file} PROPERTIES LANGUAGE CUDA)
            ENDIF()
        ENDFOREACH()

        ADD_EXECUTABLE(
            ${In_Name}
            ${ARGN})
    ELSE()
        ADD_EXECUTABLE(
            ${In_Name}
            ${ARGN})
    ENDIF()
ENDMACRO()
