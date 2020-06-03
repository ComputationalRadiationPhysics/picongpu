#
# Copyright 2014-2019 Benjamin Worpitz, Axel Huebl
#
# This file is part of Alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

IF((NOT ALPAKA_ACC_GPU_CUDA_ENABLE AND NOT ALPAKA_ACC_GPU_HIP_ENABLE) OR (ALPAKA_ACC_GPU_CUDA_ENABLE AND ALPAKA_CUDA_NVCC_SEPARABLE_COMPILATION AND ALPAKA_CUDA_COMPILER MATCHES "nvcc"))

SET(_TARGET_NAME "separableCompilation")

append_recursive_files_add_to_src_group("src/" "src/" "cpp" _FILES_SOURCE)
append_recursive_files_add_to_src_group("include/" "include/" "hpp" _FILES_HEADER)

ALPAKA_ADD_EXECUTABLE(
    ${_TARGET_NAME}
    ${_FILES_SOURCE}
    ${_FILES_HEADER})
TARGET_INCLUDE_DIRECTORIES(
    ${_TARGET_NAME}
    PRIVATE "include"
    PRIVATE ${Boost_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(
    ${_TARGET_NAME}
    PRIVATE common)

SET_TARGET_PROPERTIES(${_TARGET_NAME} PROPERTIES FOLDER "test/integ")

ADD_TEST(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} ${_ALPAKA_TEST_OPTIONS})

ENDIF()
