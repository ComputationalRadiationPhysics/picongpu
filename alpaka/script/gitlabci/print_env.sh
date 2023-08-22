#!/bin/bash

#
# Copyright 2022 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

# display output with yellow color
echo -e "\033[0;33mSteps to setup containter locally"

# display the correct docker run command
first_step_prefix="1. Run docker image via:"
if [ "${CMAKE_CXX_COMPILER}" == "nvc++" ] || [ "${alpaka_ACC_GPU_CUDA_ENABLE}" == "ON" ];
then
    if [ "${ALPAKA_CI_RUN_TESTS}" == "ON" ];
    then
	    echo "${first_step_prefix} docker run --gpus=all -it ${CI_JOB_IMAGE} bash"
    else
	    echo "${first_step_prefix} docker run -it ${CI_JOB_IMAGE} bash"
    fi
elif [ "${alpaka_ACC_GPU_HIP_ENABLE}" == "ON" ];
then
    if [ "${ALPAKA_CI_RUN_TESTS}" == "ON" ];
    then
	    echo "${first_step_prefix} docker run -it --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video ${CI_JOB_IMAGE} bash"
    else
	    echo "${first_step_prefix} docker run -it ${CI_JOB_IMAGE} bash"
    fi
else
    echo "${first_step_prefix} docker run -it ${CI_JOB_IMAGE} bash"
fi

echo -e "2. Run the following export commands in the container to setup enviroment\n"

# take all env variables, filter it and display it with a `export` prefix
printenv | grep -E 'alpaka_*|ALPAKA_*|CMAKE_*|BOOST_|CC|CXX|CUDA_' | while read -r line ; do
    echo "export $line \\"
done

echo 'export GITLAB_CI=true'
echo ""

echo "3. install git: apt update; apt install -y git"
echo "4. clone alpaka repository: git clone https://gitlab.com/hzdr/crp/alpaka.git --depth 1 -b ${CI_COMMIT_BRANCH}"
echo "5. Run the following script: cd alpaka && ./script/gitlab_ci_run.sh"
# reset the color
echo -e "\033[0m"
