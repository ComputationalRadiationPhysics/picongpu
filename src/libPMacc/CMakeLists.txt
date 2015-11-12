# Copyright 2015 Erik Zenker
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

################################################################################
# PMacc tests
################################################################################
cmake_minimum_required(VERSION 3.3)
project ("pmacc test")


################################################################################
# PMacc
################################################################################
find_package(pmacc REQUIRED CONFIG PATHS ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(SYSTEM ${pmacc_INCLUDE_DIRS})
set(LIBS ${LIBS} ${pmacc_LIBRARIES})
add_definitions(${pmacc_DEFINITIONS})

###############################################################################
# Boost.Test
###############################################################################
find_package(Boost 1.56.0 COMPONENTS unit_test_framework  REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Boost_LIBRARIES})


###############################################################################
# Targets
###############################################################################

# Test cases
file(GLOB TESTS test/*UT.cu)
cuda_add_executable(check EXCLUDE_FROM_ALL ${TESTS})
target_link_libraries(check ${LIBS})

# CTest
enable_testing()
add_test(pmacc_test_build "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target check)
add_test(pmacc_test_run  mpiexec --allow-run-as-root -n 2 check )
set_tests_properties(pmacc_test_run PROPERTIES DEPENDS graybat_check_build)
