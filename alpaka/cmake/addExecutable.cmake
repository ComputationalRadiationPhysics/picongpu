#
# Copyright 2014-2019 Benjamin Worpitz
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

CMAKE_MINIMUM_REQUIRED(VERSION 3.15)

#------------------------------------------------------------------------------
# Calls CUDA_ADD_EXECUTABLE or ADD_EXECUTABLE depending on the enabled alpaka accelerators.
# Using a macro to stay in the scope (fixes lost assignment of linker command in FindHIP.cmake)
# https://github.com/ROCm-Developer-Tools/HIP/issues/631
MACRO(ALPAKA_ADD_EXECUTABLE In_Name)
    IF(ALPAKA_ACC_GPU_CUDA_ENABLE)
        IF(ALPAKA_CUDA_COMPILER MATCHES "clang")
            FOREACH(_file ${ARGN})
                IF((${_file} MATCHES "\\.cpp$") OR (${_file} MATCHES "\\.cxx$") OR (${_file} MATCHES "\\.cu$"))
                    SET_SOURCE_FILES_PROPERTIES(${_file} PROPERTIES COMPILE_FLAGS "-x cuda")
                ENDIF()
            ENDFOREACH()
            ADD_EXECUTABLE(
                ${In_Name}
                ${ARGN})
        ELSE()
            FOREACH(_file ${ARGN})
                IF((${_file} MATCHES "\\.cpp$") OR (${_file} MATCHES "\\.cxx$"))
                    SET_SOURCE_FILES_PROPERTIES(${_file} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
                ENDIF()
            ENDFOREACH()

            SET(CUDA_LINK_LIBRARIES_KEYWORD "PUBLIC")

            CUDA_ADD_EXECUTABLE(
                ${In_Name}
                ${ARGN})
        ENDIF()
    ELSEIF(ALPAKA_ACC_GPU_HIP_ENABLE)
	      FOREACH(_file ${ARGN})
		        IF((${_file} MATCHES "\\.cpp$") OR (${_file} MATCHES "\\.cxx$"))
		            SET_SOURCE_FILES_PROPERTIES(${_file} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT OBJ)
		        ENDIF()
	      ENDFOREACH()

	      HIP_ADD_EXECUTABLE(
		        ${In_Name}
		        ${ARGN})

    ELSE()
        ADD_EXECUTABLE(
            ${In_Name}
            ${ARGN})
    ENDIF()
ENDMACRO()
