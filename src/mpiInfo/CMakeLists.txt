#
# Copyright 2013 Axel Huebl, René Widera
#
# This file is part of mpiInfo.
#
# mpiInfo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# mpiInfo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with mpiInfo.
# If not, see <http://www.gnu.org/licenses/>.
#

cmake_minimum_required(VERSION 2.8.5)


# Projekt name
project (mpiInfo)

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    SET(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "install prefix" FORCE)
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT) 


#set helper pathes to find libraries and packages
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/" "$ENV{MPI_ROOT}" "$ENV{CUDA_ROOT}" "$ENV{BOOST_ROOT}" "$ENV{HDF5_ROOT}")

FIND_PACKAGE(MPI REQUIRED)

SET(CMAKE_CXX_FLAGS_DEFAULT "-Wall")

#Boost from system
FIND_PACKAGE(Boost REQUIRED COMPONENTS program_options)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})
SET(LIBS ${LIBS} ${Boost_LIBRARIES})

INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH} ${mpi_include_path})

add_executable(mpiInfo
    main.cpp
)


target_link_libraries(mpiInfo  ${LIBS} ${MPI_LIBRARIES}) # not necessary ${MPI_EXTRA_LIBRARY}



#### INSTALLATION #####
INSTALL(TARGETS mpiInfo
         RUNTIME DESTINATION bin)
