#
# Copyright 2015-2017 Benjamin Worpitz
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

################################################################################
# Required CMake version.
################################################################################

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

PROJECT("alpakaUnitTest")

################################################################################
# Add subdirectories.
################################################################################

ADD_SUBDIRECTORY("acc/")
ADD_SUBDIRECTORY("atomic/")
ADD_SUBDIRECTORY("block/shared/")
ADD_SUBDIRECTORY("block/sync/")
ADD_SUBDIRECTORY("event/")
ADD_SUBDIRECTORY("idx/")
ADD_SUBDIRECTORY("kernel/")
ADD_SUBDIRECTORY("mem/buf/")
ADD_SUBDIRECTORY("mem/view/")
ADD_SUBDIRECTORY("meta/")
ADD_SUBDIRECTORY("rand/")
ADD_SUBDIRECTORY("stream/")
ADD_SUBDIRECTORY("time/")
ADD_SUBDIRECTORY("vec/")
