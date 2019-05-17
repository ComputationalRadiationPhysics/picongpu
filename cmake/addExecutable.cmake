#
# Copyright 2016 Rene Widera
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

# same dependency as ALPAKA_ADD_EXECUTABLE
cmake_minimum_required(VERSION 3.3.0)


macro(CUPLA_ADD_EXECUTABLE BinaryName)

    include_directories(${cupla_INCLUDE_DIRS})
    add_definitions(${cupla_DEFINITIONS})

    alpaka_add_executable(
        ${BinaryName}
        ${ARGN}
        ${cupla_SOURCE_FILES}
    )
endmacro()
