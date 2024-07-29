# Copyright 2023 Rene Widera
#
# This file is part of PIConGPU.
#
# PIConGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PIConGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.
#

cmake_minimum_required(VERSION 3.25.0)
project("UnitTest")

###############################################################################
# Language Flags
###############################################################################

# enforce C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)


################################################################################
# CMake policies
#
# Search in <PackageName>_ROOT:
#   https://cmake.org/cmake/help/v3.12/policy/CMP0074.html
################################################################################
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

################################################################################
# PMacc
################################################################################
find_package(PMacc REQUIRED CONFIG PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../../include/pmacc")

###############################################################################
# Catch2
###############################################################################

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../thirdParty/catch2 ${CMAKE_BINARY_DIR}/catch2)

###############################################################################
# JSON
###############################################################################

 set(JSON_BuildTests OFF CACHE INTERNAL "")
 set(JSON_MultipleHeaders OFF CACHE INTERNAL "")
 set(JSON_ImplicitConversions OFF CACHE INTERNAL "")
 set(JSON_Install OFF CACHE INTERNAL "")  # only used PRIVATE

add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../thirdParty/nlohmann_json"
    "${CMAKE_CURRENT_BINARY_DIR}/build_nlohmann_json")

################################################################################
# Find Boost
################################################################################

list(APPEND CMAKE_PREFIX_PATH "$ENV{BOOST_ROOT}")

find_package(Boost 1.74.0 REQUIRED COMPONENTS program_options)
if(TARGET Boost::boost)
    list(PREPEND HOST_LIBS Boost::boost Boost::program_options)
else()
    include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
    list(PREPEND HOST_LIBS ${Boost_LIBRARIES})
endif()

################################################################################
# MPI
################################################################################

# MPI is provided by pmacc but to execute the binaries via root additional flags must be given to the execution command
option(USE_MPI_AS_ROOT_USER "add --allow-run-as-root mpiexec used by ctest" OFF)

if(USE_MPI_AS_ROOT_USER)
    set(MPI_RUNTIME_FLAGS "--allow-run-as-root")
endif()

# PIConGPU

include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/../../../include")

###############################################################################
# Targets
###############################################################################

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/test)
include_directories(BEFORE ./include)

# CTest
enable_testing()

# We need to build and link this to check CLI aspects.
add_library(argparser
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../include/picongpu/ArgsParser.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../include/picongpu/versionFormat.cpp"
)

target_link_libraries(argparser PRIVATE ${HOST_LIBS})
target_link_libraries(argparser PRIVATE pmacc::pmacc)

# Test cases
# Each *UT.cpp file is an independent executable with one or more test cases
file(GLOB_RECURSE TESTS *.cpp)
foreach(dim 2 3)
    foreach(testCaseFilepath ${TESTS})
        get_filename_component(testCaseFilename ${testCaseFilepath} NAME)
        string(REPLACE "UT.cpp" "" testCase ${testCaseFilename})
        set(testExe "${PROJECT_NAME}-${testCase}-${dim}D")
        alpaka_add_executable(${testExe} ${testCaseFilepath})
        target_compile_definitions(${testExe} PRIVATE TEST_DIM=${dim})
        target_link_libraries(${testExe} PUBLIC Catch2 Catch2WithMain)
        target_link_libraries(${testExe} PRIVATE pmacc::pmacc)
        if(${testExe} MATCHES ".*CLI.*")
          target_link_libraries(${testExe} PUBLIC argparser)
        endif()
        target_include_directories(
            ${testExe}
            SYSTEM PRIVATE
            $<TARGET_PROPERTY:nlohmann_json::nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>)
        target_link_libraries(${testExe} PRIVATE nlohmann_json::nlohmann_json)
        add_test(NAME "${testCase}-${dim}D" COMMAND mpiexec ${MPI_RUNTIME_FLAGS} -n 1 ./${testExe})
    endforeach()
    string(REPLACE "-DTEST_DIM=${dim}" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endforeach()
