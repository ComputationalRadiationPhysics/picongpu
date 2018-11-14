#!/bin/bash

#
# Copyright 2017-2018 Benjamin Worpitz
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

: ${ALPAKA_ACC_GPU_CUDA_ENABLE?"ALPAKA_ACC_GPU_CUDA_ENABLE must be specified"}
: ${ALPAKA_ACC_GPU_HIP_ENABLE?"ALPAKA_ACC_GPU_HIP_ENABLE must be specified"}

# https://stackoverflow.com/questions/42218009/how-to-tell-if-any-command-in-bash-script-failed-non-zero-exit-status
err=0
trap 'err=1' ERR

if [ "${ALPAKA_ACC_GPU_CUDA_ENABLE}" == "OFF" ] && [ "${ALPAKA_ACC_GPU_HIP_ENABLE}" == "OFF" ];
then
    cd build/make/

    ctest -V

    cd ../..
fi

test $err = 0 # Return non-zero if any command failed
