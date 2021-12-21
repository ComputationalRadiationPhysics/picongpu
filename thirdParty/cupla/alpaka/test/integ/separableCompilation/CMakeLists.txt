#
# Copyright 2014-2020 Benjamin Worpitz, Axel Huebl, Jan Stephan
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

if((NOT ALPAKA_ACC_GPU_CUDA_ENABLE AND NOT ALPAKA_ACC_GPU_HIP_ENABLE
   AND NOT ALPAKA_ACC_ANY_BT_OMP5_ENABLE AND NOT ALPAKA_ACC_ANY_BT_OACC_ENABLE) OR
   (ALPAKA_ACC_GPU_CUDA_ENABLE AND ALPAKA_CUDA_NVCC_SEPARABLE_COMPILATION AND ALPAKA_CUDA_COMPILER MATCHES "nvcc"))

    set(_TARGET_NAME "separableCompilation")

    append_recursive_files_add_to_src_group("src/" "src/" "cpp" _FILES_SOURCE)
    append_recursive_files_add_to_src_group("include/" "include/" "hpp" _FILES_HEADER)

    alpaka_add_executable(
        ${_TARGET_NAME}
        ${_FILES_SOURCE}
        ${_FILES_HEADER})
    target_include_directories(
        ${_TARGET_NAME}
        PRIVATE "include")
    target_link_libraries(
        ${_TARGET_NAME}
        PRIVATE common)

    set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER "test/integ")

    add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} ${_ALPAKA_TEST_OPTIONS})

endif()
