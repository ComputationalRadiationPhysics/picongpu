#!/bin/bash

#
# Copyright 2014-2021 Benjamin Worpitz, Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#
set +xv
source ./script/setup_utilities.sh

echo_green "<SCRIPT: run_build>"

cd build/

if [[ -z "${ALPAKA_CI_BUILD_JOBS+x}" ]]
then
    # on Windows and MacOS asking for number of threads and memory is not implemented
    # therefore we cannot calculate the optimal number of build process
    if [[ "$ALPAKA_CI_OS_NAME" = "Windows" ]] || [[ "$ALPAKA_CI_OS_NAME" = "macOS" ]]; then
        echo_yellow "ALPAKA_CI_BUILD_JOBS is not set and OS is ${ALPAKA_CI_OS_NAME}. Use fallback."
        ALPAKA_CI_BUILD_JOBS=1
    else
        if [[ -z "${ALPAKA_CI_REQUIRED_RAM_PER_BUILD_THREAD_BYTES+x}" ]]; then
            # fallback
            echo_yellow "ALPAKA_CI_BUILD_JOBS and ALPAKA_CI_REQUIRED_RAM_PER_BUILD_THREAD_BYTES" \
                "is not set and OS is ${ALPAKA_CI_OS_NAME}. Use fallback."
            ALPAKA_CI_BUILD_JOBS=1
        else
            if [[ -n ${GITHUB_ACTIONS+x} ]]; then
                max_num_build_threads=$(nproc)
                total_memory_bytes=$(free -b | awk '/Mem:/ { print $2 }')
            elif [[ -n ${GITLAB_CI+x} ]]; then
                # CI_CPU and CI_RAM_BYTES_TOTAL are predefined on the HZDR runner
                max_num_build_threads="${CI_CPUS}"
                total_memory_bytes="${CI_RAM_BYTES_TOTAL}"
            else
                # local container
                max_num_build_threads=$(nproc)
                total_memory_bytes=$(free -b | awk '/Mem:/ { print $2 }')
            fi

            ALPAKA_CI_BUILD_JOBS=$(($total_memory_bytes / ALPAKA_CI_REQUIRED_RAM_PER_BUILD_THREAD_BYTES))
            if [[ $max_num_build_threads -le $ALPAKA_CI_BUILD_JOBS ]]; then
                ALPAKA_CI_BUILD_JOBS=$max_num_build_threads
            fi

            total_memory_mb=$((total_memory_bytes / 1024 / 1024))

            echo_green "Calculate optimal number of threads:" \
            "${ALPAKA_CI_BUILD_JOBS}\n" \
            "Available maximum number of threads: ${max_num_build_threads}\n" \
            "Available number of memory: ${total_memory_mb} MB"
        fi
    fi
else
    echo_green "ALPAKA_CI_BUILD_JOBS is set to ${ALPAKA_CI_BUILD_JOBS}"
fi

if [ "$ALPAKA_CI_OS_NAME" = "Linux" ] || [ "$ALPAKA_CI_OS_NAME" = "macOS" ]
then
    make VERBOSE=1 -j${ALPAKA_CI_BUILD_JOBS}
elif [ "$ALPAKA_CI_OS_NAME" = "Windows" ]
then
    "$MSBUILD_EXECUTABLE" "alpaka.sln" -p:Configuration=${CMAKE_BUILD_TYPE} -maxcpucount:${ALPAKA_CI_BUILD_JOBS} -verbosity:minimal
fi

cd ..
