#
# Copyright 2014-2018 Benjamin Worpitz
#
# This file is part of alpaka.
#
# alpaka is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# alpaka is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with alpaka.
# If not, see <http://www.gnu.org/licenses/>.
#

SET(_TARGET_NAME "matMul")

append_recursive_files_add_to_src_group("src/" "src/" "cpp" _FILES_SOURCE)

ALPAKA_ADD_EXECUTABLE(
    ${_TARGET_NAME}
    ${_FILES_SOURCE})
TARGET_INCLUDE_DIRECTORIES(
    ${_TARGET_NAME}
    PRIVATE ${Boost_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(
    ${_TARGET_NAME}
    PRIVATE common ${Boost_LIBRARIES})
IF(NOT Boost_USE_STATIC_LIBS)
    TARGET_COMPILE_DEFINITIONS(
        ${_TARGET_NAME}
        PRIVATE "BOOST_TEST_DYN_LINK")
ENDIF()

SET_TARGET_PROPERTIES(${_TARGET_NAME} PROPERTIES FOLDER "test/integ")

ADD_TEST(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} --log_level=test_suite --color_output=true)
