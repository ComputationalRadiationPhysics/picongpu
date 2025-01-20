# Copyright 2024 Benjamin Worpitz, Jakob Krude, Andrea Bocci
# SPDX-License-Identifier: MPL-2.0

set(_TARGET_NAME "execTest")

append_recursive_files_add_to_src_group("src/" "src/" "cpp" _FILES_SOURCE)
append_recursive_files_add_to_src_group("src/" "src/" "hpp" _FILES_HEADER)

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

add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} ${_alpaka_TEST_OPTIONS})
