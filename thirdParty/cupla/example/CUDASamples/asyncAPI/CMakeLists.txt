#
# Copyright 2016-2021 Rene Widera, Benjamin Worpitz, Simeon Ehrig
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

################################################################################
# Required CMake version.
################################################################################

cmake_minimum_required(VERSION 3.22.0)

################################################################################
# Project.
################################################################################

project(asyncAPI)

################################################################################
# Find cupla
################################################################################

if(NOT TARGET cupla::cupla)
  find_package(cupla REQUIRED)
endif()

################################################################################
# Add library.
################################################################################

cupla_add_executable(${PROJECT_NAME} "src/asyncAPI.cpp")
target_include_directories(${PROJECT_NAME} PRIVATE "../common/")
