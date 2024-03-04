#
# Copyright 2014-2020 Benjamin Worpitz, Axel Huebl, Jan Stephan
# SPDX-License-Identifier: MPL-2.0
#

set(_TARGET_NAME "axpyTest")

append_recursive_files_add_to_src_group("src/" "src/" "cpp" _FILES_SOURCE)

alpaka_add_executable(
    ${_TARGET_NAME}
    ${_FILES_SOURCE})
target_link_libraries(
    ${_TARGET_NAME}
    PRIVATE common)

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER "test/integ")

add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} ${_alpaka_TEST_OPTIONS})
