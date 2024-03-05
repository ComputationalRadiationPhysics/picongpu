# Copyright 2023 Benjamin Worpitz, Jan Stephan, Antonio Di Pilato
# SPDX-License-Identifier: MPL-2.0
#

set(_TARGET_NAME "coreTest")

append_recursive_files_add_to_src_group("src/" "src/" "cpp" _FILES_SOURCE)

# TODO(bgruber): Revisit this in the future on a new CI image. This problem does not happen locally.
if(CMAKE_CXX_FLAGS MATCHES ".*-fsanitize=thread.*")
    message(WARNING "Part of the threadpool test fails the TSan CI and is therefore disabled. See also: https://github.com/alpaka-group/alpaka/issues/2101")
    set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/ThreadPool.cpp" PROPERTIES COMPILE_DEFINITIONS "ALPAKA_USES_TSAN")
endif()

alpaka_add_executable(
    ${_TARGET_NAME}
    ${_FILES_SOURCE})
target_link_libraries(
    ${_TARGET_NAME}
    PRIVATE common)

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER "test/unit")

add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} ${_alpaka_TEST_OPTIONS})
