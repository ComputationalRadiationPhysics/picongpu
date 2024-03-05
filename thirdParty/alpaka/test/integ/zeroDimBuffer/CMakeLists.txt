#
# Copyright 2021 Andrea Bocci
# SPDX-License-Identifier: MPL-2.0
#

set(_TARGET_NAME "zeroDimBufferTest")

append_recursive_files_add_to_src_group("src/" "src/" "cpp" _FILES_SOURCE)

alpaka_add_executable(
    ${_TARGET_NAME}
    ${_FILES_SOURCE})
target_link_libraries(
    ${_TARGET_NAME}
    PRIVATE common)

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER "test/integ")

if(alpaka_ACC_GPU_CUDA_ENABLE OR alpaka_ACC_GPU_HIP_ENABLE)
    add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} ${_alpaka_TEST_OPTIONS})
endif()
