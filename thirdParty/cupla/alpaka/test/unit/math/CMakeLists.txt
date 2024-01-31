#
# Copyright 2017-2019 Benjamin Worpitz, Jakob Krude
# SPDX-License-Identifier: MPL-2.0
#

set(_TARGET_NAME "mathTest")

append_recursive_files_add_to_src_group("src/" "src/" "cpp" _FILES_SOURCE)
append_recursive_files_add_to_src_group("src/" "src/" "hpp" _FILES_HEADER)

if(alpaka_ACC_GPU_HIP_ENABLE)
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

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER "test/unit")
target_compile_definitions(${_TARGET_NAME} PRIVATE "-DTEST_UNIT_MATH")

add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} ${_alpaka_TEST_OPTIONS})
