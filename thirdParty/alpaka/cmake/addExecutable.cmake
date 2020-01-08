#
# Copyright 2014-2019 Benjamin Worpitz
#
# This file is part of Alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

CMAKE_MINIMUM_REQUIRED(VERSION 3.11.4)

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
            IF (CMAKE_VERSION VERSION_LESS 3.9.0)
                CMAKE_POLICY(SET CMP0023 OLD)   # CUDA_ADD_EXECUTABLE calls TARGET_LINK_LIBRARIES without keywords.
            ELSE()
                SET(CUDA_LINK_LIBRARIES_KEYWORD "PUBLIC")
            ENDIF()
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
        IF (CMAKE_VERSION VERSION_LESS 3.9.0)
            CMAKE_POLICY(SET CMP0023 OLD)   # CUDA_ADD_EXECUTABLE calls TARGET_LINK_LIBRARIES without keywords.
        ELSE()
            SET(HIP_LINK_LIBRARIES_KEYWORD "PUBLIC")
        ENDIF()

	      HIP_ADD_EXECUTABLE(
		        ${In_Name}
		        ${ARGN})

    ELSE()
        ADD_EXECUTABLE(
            ${In_Name}
            ${ARGN})
    ENDIF()
ENDMACRO()
