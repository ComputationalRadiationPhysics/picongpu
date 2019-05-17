#
# Copyright 2014-2019 Benjamin Worpitz
#
# This file is part of Alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

################################################################################
# Required CMake version.
################################################################################

CMAKE_MINIMUM_REQUIRED(VERSION 3.11.0)

PROJECT("alpakaIntegTest")

################################################################################
# Add subdirectories.
################################################################################

ADD_SUBDIRECTORY("axpy/")
ADD_SUBDIRECTORY("cudaOnly/")
ADD_SUBDIRECTORY("mandelbrot/")
ADD_SUBDIRECTORY("matMul/")
ADD_SUBDIRECTORY("sharedMem/")
