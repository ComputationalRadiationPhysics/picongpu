#
# Copyright 2023 Benjamin Worpitz, Jan Stephan
# SPDX-License-Identifier: ISC
#

################################################################################
# Required CMake version.
################################################################################

cmake_minimum_required(VERSION 3.22)

project("alpakaExamples" LANGUAGES CXX)

################################################################################
# Add subdirectories.
################################################################################

add_subdirectory("bufferCopy/")
add_subdirectory("complex/")
add_subdirectory("counterBasedRng/")
add_subdirectory("heatEquation/")
add_subdirectory("helloWorld/")
add_subdirectory("helloWorldLambda/")
add_subdirectory("kernelSpecialization/")
add_subdirectory("ls/")
add_subdirectory("monteCarloIntegration/")
add_subdirectory("openMPSchedule/")
add_subdirectory("parallelLoopPatterns/")
add_subdirectory("randomStrategies/")
add_subdirectory("randomCells2D/")
add_subdirectory("reduce/")
add_subdirectory("tagSpecialization/")
add_subdirectory("vectorAdd/")
add_subdirectory("babelstream/")
