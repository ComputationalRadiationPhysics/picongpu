#
# Copyright 2016 Rene Widera, Benjamin Worpitz
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

find_path(
    _cupla_ROOT_DIR
    NAMES "include/cuda_to_cupla.hpp"
    HINTS "${cupla_ROOT}" ENV CUPLA_ROOT
    DOC "cupla ROOT location")

if(_cupla_ROOT_DIR)
    include("${_cupla_ROOT_DIR}/cuplaConfig.cmake")
else()
    message(FATAL_ERROR "cupla could not be found!")
endif()
