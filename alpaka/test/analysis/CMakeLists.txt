#
# Copyright 2023 Benjamin Worpitz, Jan Stephan
# SPDX-License-Identifier: MPL-2.0
#

################################################################################
# Required CMake version.
################################################################################

cmake_minimum_required(VERSION 3.22)

project("alpakaAnalysisTest" LANGUAGES CXX)

################################################################################
# Add subdirectories.
################################################################################

add_subdirectory("headerCheck/")
