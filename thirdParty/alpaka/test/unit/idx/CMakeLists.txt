#
# Copyright 2017-2019 Benjamin Worpitz, Axel Huebl
#
# This file is part of Alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

SET(_TARGET_NAME "idx")

append_recursive_files_add_to_src_group("src/" "src/" "cpp" _FILES_SOURCE)

ALPAKA_ADD_EXECUTABLE(
    ${_TARGET_NAME}
    ${_FILES_SOURCE})
TARGET_INCLUDE_DIRECTORIES(
    ${_TARGET_NAME}
    PRIVATE ${Boost_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(
    ${_TARGET_NAME}
    PRIVATE common)

SET_TARGET_PROPERTIES(${_TARGET_NAME} PROPERTIES FOLDER "test/unit")

ADD_TEST(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} ${_ALPAKA_TEST_OPTIONS})
