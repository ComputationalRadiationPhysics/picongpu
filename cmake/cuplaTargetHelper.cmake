#
# Copyright 2021 Simeon Ehrig
#
# This file is part of cupla.
#
# cupla is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cupla is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with cupla.
# If not, see <http://www.gnu.org/licenses/>.
#

# creates the cupla target (library)
macro(createCuplaTarget
    TARGET_NAME # name of the cupla target, normally cupla
    _CUPLA_INCLUDE_DIR # path of the include folder
    _CUPLA_SRC_DIR # path of the src folder
    )
  alpaka_add_library(
    ${TARGET_NAME}
    STATIC
    ${_CUPLA_SRC_DIR}/manager/Driver.cpp
    ${_CUPLA_SRC_DIR}/common.cpp
    ${_CUPLA_SRC_DIR}/device.cpp
    ${_CUPLA_SRC_DIR}/event.cpp
    ${_CUPLA_SRC_DIR}/memory.cpp
    ${_CUPLA_SRC_DIR}/stream.cpp
    )

  target_include_directories(${TARGET_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${_CUPLA_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>)

  # Even if there are no sources CMAKE has to know the language.
  set_target_properties(${TARGET_NAME} PROPERTIES LINKER_LANGUAGE CXX)

  target_link_libraries(
    ${TARGET_NAME}
    PUBLIC
    alpaka::alpaka)

  if(CUPLA_STREAM_ASYNC_ENABLE)
    TARGET_COMPILE_DEFINITIONS(${TARGET_NAME} PUBLIC "CUPLA_STREAM_ASYNC_ENABLED=1")
  else()
    TARGET_COMPILE_DEFINITIONS(${TARGET_NAME} PUBLIC "CUPLA_STREAM_ASYNC_ENABLED=0")
  endif()

  if(NOT(ALPAKA_ACC_GPU_CUDA_ENABLE OR ALPAKA_ACC_GPU_HIP_ENABLE))
    # GNU
    if(CMAKE_COMPILER_IS_GNUCXX)
      TARGET_COMPILE_OPTIONS(${TARGET_NAME}
        PRIVATE
        "-Wall"
        "-Wextra"
        "-Wno-unknown-pragmas"
        "-Wno-unused-parameter"
        "-Wno-unused-local-typedefs"
        "-Wno-attributes"
        "-Wno-reorder"
        "-Wno-sign-compare")
      # ICC
    elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
      TARGET_COMPILE_OPTIONS(${TARGET_NAME} PRIVATE "-Wall")
      # PGI
    elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "PGI")
      TARGET_COMPILE_OPTIONS(${TARGET_NAME} PRIVATE "-Minform=inform")
    endif()
  endif()

  add_library(${TARGET_NAME}::${TARGET_NAME} ALIAS ${TARGET_NAME})
endmacro()
