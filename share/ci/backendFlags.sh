#!/bin/bash

###################################################
# translate PIConGPU backend names into CMake Flags
###################################################

# CUDA 13 dropped offline compilation for Maxwell, Pascal and Volta, i.e. for
# compute capabilities below Turing (sm_75), so sm_60 is not buildable anymore
# with those toolkits. Older toolkits keep using sm_60, which is PIConGPU's
# minimum compute capability, and it stays the fallback when the toolkit
# version can not be detected.
# https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#deprecated-architectures
get_default_cuda_arch()
{
    cuda_major=$( (nvcc --version 2>/dev/null || true) \
        | sed -n 's/.*release \([0-9]\{1,\}\)\..*/\1/p' | head -n 1 )
    if [ -n "$cuda_major" ] && [ "$cuda_major" -ge 13 ] ; then
        echo 75
    else
        echo 60
    fi
}

get_backend_flags()
{
    backend_cfg=(${1//:/ })
    num_options="${#backend_cfg[@]}"
    if [ $num_options -gt 2 ] ; then
        echo "-b|--backend must be contain 'backend:arch' or 'backend'" >&2
        exit 1
    fi
    if [ "${backend_cfg[0]}" == "cuda" ] ; then
        result+=" -Dalpaka_ACC_GPU_CUDA_ENABLE=ON -Dalpaka_ACC_GPU_CUDA_ONLY_MODE=ON"
        if [ $num_options -eq 2 ] ; then
            result+=" -DCMAKE_CUDA_ARCHITECTURES=${backend_cfg[1]}"
        else
            result+=" -DCMAKE_CUDA_ARCHITECTURES=$(get_default_cuda_arch)"
        fi
    elif [ "${backend_cfg[0]}" == "omp2b" ] ; then
        result+=" -Dalpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE=ON"
        if [ $num_options -eq 2 ] ; then
            result+=" -DPMACC_CPU_ARCH=${backend_cfg[1]}"
        fi
    elif [ "${backend_cfg[0]}" == "serial" ] ; then
        result+=" -Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON"
        if [ $num_options -eq 2 ] ; then
            result+=" -DPMACC_CPU_ARCH=${backend_cfg[1]}"
        fi
    elif [ "${backend_cfg[0]}" == "tbb" ] ; then
        result+=" -Dalpaka_ACC_CPU_B_TBB_T_SEQ_ENABLE=ON"
        if [ $num_options -eq 2 ] ; then
            result+=" -DPMACC_CPU_ARCH=${backend_cfg[1]}"
        fi
    elif [ "${backend_cfg[0]}" == "threads" ] ; then
        result+=" -Dalpaka_ACC_CPU_B_SEQ_T_THREADS_ENABLE=ON"
        if [ $num_options -eq 2 ] ; then
            result+=" -DPMACC_CPU_ARCH=${backend_cfg[1]}"
        fi
    elif [ "${backend_cfg[0]}" == "hip" ] ; then
        result+=" -Dalpaka_ACC_GPU_HIP_ENABLE=ON -Dalpaka_ACC_GPU_HIP_ONLY_MODE=ON"
        if [ $num_options -eq 2 ] ; then
            result+=" -DCMAKE_HIP_ARCHITECTURES=${backend_cfg[1]}"
        else
            # If no architecture is given build for Radeon VII or MI50/60.
            result+=" -DCMAKE_HIP_ARCHITECTURES=gfx906"
        fi
    else
        echo "unsupported backend given '$1'" >&2
        exit 1
    fi

    echo "$result"
    exit 0
}
