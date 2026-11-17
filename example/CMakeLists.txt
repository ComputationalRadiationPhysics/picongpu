#
# Copyright 2026 Benjamin Worpitz, Jan Stephan
# SPDX-License-Identifier: ISC
#

################################################################################
# Required CMake version.
################################################################################

cmake_minimum_required(VERSION 3.25)

project("alpakaExamples" LANGUAGES CXX)

################################################################################
# Add subdirectories.
################################################################################

add_subdirectory("bufferCopy/")
add_subdirectory("complex/")
add_subdirectory("convolution1D/")
add_subdirectory("convolution2D/")
add_subdirectory("conv2DWithMdspan/")
add_subdirectory("counterBasedRng/")
add_subdirectory("heatEquation/")
add_subdirectory("heatEquation2D/")
add_subdirectory("helloWorld/")
add_subdirectory("helloWorldLambda/")
add_subdirectory("helloWorldGridSync/")
add_subdirectory("kernelSpecialization/")
add_subdirectory("ls/")
add_subdirectory("matrixMulWithMdspan/")
add_subdirectory("monteCarloIntegration/")
add_subdirectory("openMPSchedule/")
add_subdirectory("parallelLoopPatterns/")
add_subdirectory("randomStrategies/")
add_subdirectory("randomCells2D/")
add_subdirectory("reduce/")
add_subdirectory("tagSpecialization/")
add_subdirectory("useBLASInAlpaka/useCuBLASInAlpaka/")
add_subdirectory("useBLASInAlpaka/useRocBLASInAlpaka/")
add_subdirectory("vectorAdd/")
