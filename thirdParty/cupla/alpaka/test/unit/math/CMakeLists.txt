#
# Copyright 2017-2019 Benjamin Worpitz, Jakob Krude
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

set(_TARGET_NAME "math")

append_recursive_files_add_to_src_group("src/" "src/" "cpp" _FILES_SOURCE)
append_recursive_files_add_to_src_group("src/" "src/" "hpp" _FILES_HEADER)

if(ALPAKA_ACC_GPU_CUDA_ENABLE)
    list(REMOVE_ITEM
            CUDA_NVCC_FLAGS "--ftz=true" "--prec-div=false" "--prec-sqrt=false" "--fmad=true" "--use_fast_math" "-use_fast_math")

endif()
if(ALPAKA_ACC_GPU_HIP_ENABLE)
    list(REMOVE_ITEM
            HIP_NVCC_FLAGS "--ftz=true" "--prec-div=false" "--prec-sqrt=false" "--fmad=true" "--use_fast_math" "-use_fast_math")
endif()

alpaka_add_executable(
    ${_TARGET_NAME}
    ${_FILES_SOURCE}
    ${_FILES_HEADER})
target_include_directories(
    ${_TARGET_NAME}
    PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(
    ${_TARGET_NAME}
    PRIVATE common)
set_target_properties(
    ${_TARGET_NAME}
    PROPERTIES
    COMPILE_OPTIONS
    $<$<CXX_COMPILER_ID:clang>:"-ffp-contract=off"> # ffp-contract: https://llvm.org/docs/CompileCudaWithLLVM.html#id5
    )

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER "test/unit")
target_compile_definitions(${_TARGET_NAME} PRIVATE "-DTEST_UNIT_MATH")

add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} ${_ALPAKA_TEST_OPTIONS})
