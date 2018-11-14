#
# Copyright 2014-2016 Benjamin Worpitz
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
