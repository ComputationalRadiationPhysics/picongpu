#
# Copyright 2015-2016 Erik Zenker, Alexander Grund
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
cmake_minimum_required(VERSION 3.3.0)
project(PMaccTest)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 11)

################################################################################
# PMacc
################################################################################
find_package(PMacc REQUIRED CONFIG PATHS ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(SYSTEM ${PMacc_INCLUDE_DIRS})
set(LIBS ${LIBS} ${PMacc_LIBRARIES})
add_definitions(${PMacc_DEFINITIONS})

###############################################################################
# Boost.Test
###############################################################################
find_package(Boost 1.57.0 COMPONENTS unit_test_framework  REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Boost_LIBRARIES})


###############################################################################
# Targets
###############################################################################

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/test)
add_definitions(-DBOOST_TEST_DYN_LINK)

# CTest
enable_testing()

# Test cases
# Each *UT.cu file is an independent executable with one or more test cases
file(GLOB_RECURSE TESTS test/*UT.cu)
foreach(dim 2 3)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTEST_DIM=${dim}")
    foreach(testCaseFilepath ${TESTS})
        get_filename_component(testCaseFilename ${testCaseFilepath} NAME)
        string(REPLACE "UT.cu" "" testCase ${testCaseFilename})
        set(testExe "${PROJECT_NAME}-${testCase}-${dim}D")
        cuda_add_executable(${testExe} ${testCaseFilepath} ${CMAKE_CURRENT_SOURCE_DIR}/test/main.cpp)
        target_link_libraries(${testExe} ${LIBS})
        add_test(NAME "${testCase}-${dim}D" COMMAND mpiexec -n 1 ./${testExe})
    endforeach()
    string(REPLACE "-DTEST_DIM=${dim}" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endforeach()
