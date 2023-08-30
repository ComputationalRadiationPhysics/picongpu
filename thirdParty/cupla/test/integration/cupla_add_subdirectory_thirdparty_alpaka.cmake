#
# Copyright 2021 Simeon Ehrig
#
# This file is part of cupla.
#
# cupla is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cupla is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with cupla.
# If not, see <http://www.gnu.org/licenses/>.
#

cmake_minimum_required(VERSION 3.22.0)
project(cuplaVectorAdd)

add_subdirectory(alpaka)
add_subdirectory(cupla)

cupla_add_executable(${PROJECT_NAME} ${CMAKE_CURRENT_LIST_DIR}/vectorAdd.cpp)

install(TARGETS ${PROJECT_NAME})
