#!/bin/bash

#
# Copyright 2017 Benjamin Worpitz
#
# This file is part of alpaka.
#
# alpaka is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# alpaka is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with alpaka.
# If not, see <http://www.gnu.org/licenses/>.
#

#-------------------------------------------------------------------------------
# v: all lines are printed before executing them.
set -vuo pipefail

: ${ALPAKA_ACC_GPU_CUDA_ONLY_MODE?"ALPAKA_ACC_GPU_CUDA_ONLY_MODE must be specified"}

# https://stackoverflow.com/questions/42218009/how-to-tell-if-any-command-in-bash-script-failed-non-zero-exit-status
err=0
trap 'err=1' ERR

#-------------------------------------------------------------------------------
# Build and execute all unit tests.
BOOST_TEST_OPTIONS="--log_level=test_suite --color_output=true"
./script/travis/compileExec.sh "test/unit/acc/" "./acc ${BOOST_TEST_OPTIONS}"
./script/travis/compileExec.sh "test/unit/atomic/" "./atomic ${BOOST_TEST_OPTIONS}"
./script/travis/compileExec.sh "test/unit/block/shared/" "./blockShared ${BOOST_TEST_OPTIONS}"
./script/travis/compileExec.sh "test/unit/block/sync/" "./blockSync ${BOOST_TEST_OPTIONS}"
./script/travis/compileExec.sh "test/unit/event/" "./event ${BOOST_TEST_OPTIONS}"
./script/travis/compileExec.sh "test/unit/idx/" "./idx ${BOOST_TEST_OPTIONS}"
./script/travis/compileExec.sh "test/unit/kernel/" "./kernel ${BOOST_TEST_OPTIONS}"
./script/travis/compileExec.sh "test/unit/mem/buf/" "./memBuf ${BOOST_TEST_OPTIONS}"
./script/travis/compileExec.sh "test/unit/mem/view/" "./memView ${BOOST_TEST_OPTIONS}"
./script/travis/compileExec.sh "test/unit/meta/" "./meta ${BOOST_TEST_OPTIONS}"
./script/travis/compileExec.sh "test/unit/rand/" "./rand ${BOOST_TEST_OPTIONS}"
./script/travis/compileExec.sh "test/unit/stream/" "./stream ${BOOST_TEST_OPTIONS}"
./script/travis/compileExec.sh "test/unit/time/" "./time ${BOOST_TEST_OPTIONS}"
./script/travis/compileExec.sh "test/unit/vec/" "./vec ${BOOST_TEST_OPTIONS}"

#-------------------------------------------------------------------------------
# Build and execute all integration tests.
./script/travis/compileExec.sh "test/integ/axpy/" ./axpy
if [ "${ALPAKA_ACC_GPU_CUDA_ONLY_MODE}" == "ON" ] ;then ./script/travis/compileExec.sh "test/integ/cudaOnly/" ./cudaOnly ;fi
./script/travis/compileExec.sh "test/integ/mandelbrot/" ./mandelbrot
./script/travis/compileExec.sh "test/integ/matMul/" ./matMul
./script/travis/compileExec.sh "test/integ/sharedMem/" ./sharedMem

#-------------------------------------------------------------------------------
# Build and execute all examples.
# NOTE: The examples are hard-coded to use a CPU accelerator which is not available in ALPAKA_ACC_GPU_CUDA_ONLY_MODE
if [ "${ALPAKA_ACC_GPU_CUDA_ONLY_MODE}" == "OFF" ]
then
    ./script/travis/compileExec.sh "example/bufferCopy" ./bufferCopy
    ./script/travis/compileExec.sh "example/helloWorld" ./helloWorld
    ./script/travis/compileExec.sh "example/helloWorldLambda" ./helloWorldLambda
    ./script/travis/compileExec.sh "example/vectorAdd" ./vectorAdd
fi

test $err = 0 # Return non-zero if any command failed
