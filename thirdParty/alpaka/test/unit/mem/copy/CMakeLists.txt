#
# Copyright 2014-2020 Benjamin Worpitz, Axel Huebl
# SPDX-License-Identifier: MPL-2.0
#

set(_TARGET_NAME "bufSlicingTest")

append_recursive_files_add_to_src_group("src/" "src/" "cpp" _FILES_SOURCE)

alpaka_add_executable(
        ${_TARGET_NAME}
        ${_FILES_SOURCE})
target_link_libraries(
        ${_TARGET_NAME}
        PRIVATE common)

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER "test/unit")

if (CMAKE_CXX_COMPILER_ID STREQUAL "PGI" OR CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    target_compile_options(${_TARGET_NAME} PRIVATE "-Wc,--pending_instantiations=196")
    # warning: #185-D: dynamic initialization in unreachable code
    #     TestContainer<Dim, Idx, Acc, Data> slicingTest;
    target_compile_options(${_TARGET_NAME} PRIVATE "--diag_suppress 185")
endif()

add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} ${_alpaka_TEST_OPTIONS})
