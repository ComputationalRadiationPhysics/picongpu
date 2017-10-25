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
# e: exit as soon as one command returns a non-zero exit code.
set -e

#run tests
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI=${ALPAKA_CI}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_DEBUG=${ALPAKA_DEBUG}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_ACC_GPU_CUDA_ONLY_MODE=${ALPAKA_ACC_GPU_CUDA_ONLY_MODE}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

# If we have created the image in the current run, we do not have to load it again, because it is already available.
if [ "${ALPAKA_DOCKER_BUILD_REQUIRED}" -eq 0 ]
then
    gzip -dc "${ALPAKA_CI_DOCKER_CACHE_IMAGE_FILE_PATH}" | docker load
fi

docker run -v "$(pwd)":"$(pwd)" -w "$(pwd)" "${ALPAKA_DOCKER_ENV_LIST[@]}" --rm "${ALPAKA_CI_DOCKER_IMAGE_NAME}" /bin/bash ./script/travis/run.sh
