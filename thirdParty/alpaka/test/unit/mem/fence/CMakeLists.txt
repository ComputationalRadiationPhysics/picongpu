#
# Copyright 2023 Jan Stephan, Bernhard Manfred Gruber
# SPDX-License-Identifier: MPL-2.0
#

if (CMAKE_CXX_FLAGS MATCHES ".*-fsanitize=thread.*")
    message(WARNING "TSan does not support memory fences yet. The fenceTest has been disabled. See also: https://github.com/google/sanitizers/issues/1352")
    return()
endif()

set(_TARGET_NAME "fenceTest")

append_recursive_files_add_to_src_group("src/" "src/" "cpp" _FILES_SOURCE)

alpaka_add_executable(
        ${_TARGET_NAME}
        ${_FILES_SOURCE})

target_link_libraries(
        ${_TARGET_NAME}
        PRIVATE common)

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER "test/unit")
target_compile_definitions(${_TARGET_NAME} PRIVATE "-DTEST_UNIT_FENCE")

add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} ${_alpaka_TEST_OPTIONS})
