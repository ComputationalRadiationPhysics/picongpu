# Copyright 2016 Alexander Grund
#
# This file is part of libPMacc.
#
# libPMacc is free software: you can redistribute it and/or modify
# it under the terms of either the GNU General Public License or
# the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# libPMacc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License and the GNU Lesser General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# and the GNU Lesser General Public License along with libPMacc.
# If not, see <http://www.gnu.org/licenses/>.
#

cmake_minimum_required(VERSION 3.3)
project("TestRandomGenerators")

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/../..")

################################################################################
# PMacc
################################################################################
find_package(PMacc REQUIRED CONFIG PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../..")
include_directories(SYSTEM ${PMacc_INCLUDE_DIRS})
set(LIBS ${LIBS} ${PMacc_LIBRARIES})
add_definitions(${PMacc_DEFINITIONS})

###############################################################################
# Targets
###############################################################################

cuda_add_executable(TestRng 2DDistribution.cu)
target_link_libraries(TestRng ${LIBS})

add_custom_target(run
    COMMAND mpiexec -n 1 TestRng
    DEPENDS TestRng
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
