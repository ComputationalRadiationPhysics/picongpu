#!/bin/bash

###################################################
# translate PIConGPU backend names into CMake Flags
###################################################

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
            result+=" -DCMAKE_CUDA_ARCHITECTURES=\"${backend_cfg[1]}\""
        else
            result+=" -DCMAKE_CUDA_ARCHITECTURES=52"
        fi
    elif [ "${backend_cfg[0]}" == "omp2b" ] ; then
        result+=" -Dalpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE=ON"
        if [ $num_options -eq 2 ] ; then
            result+=" -DPMACC_CPU_ARCH=\"${backend_cfg[1]}\""
        fi
    elif [ "${backend_cfg[0]}" == "serial" ] ; then
        result+=" -Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON"
        if [ $num_options -eq 2 ] ; then
            result+=" -DPMACC_CPU_ARCH=\"${backend_cfg[1]}\""
        fi
    elif [ "${backend_cfg[0]}" == "tbb" ] ; then
        result+=" -Dalpaka_ACC_CPU_B_TBB_T_SEQ_ENABLE=ON"
        if [ $num_options -eq 2 ] ; then
            result+=" -DPMACC_CPU_ARCH=\"${backend_cfg[1]}\""
        fi
    elif [ "${backend_cfg[0]}" == "threads" ] ; then
        result+=" -Dalpaka_ACC_CPU_B_SEQ_T_THREADS_ENABLE=ON"
        if [ $num_options -eq 2 ] ; then
            result+=" -DPMACC_CPU_ARCH=\"${backend_cfg[1]}\""
        fi
    elif [ "${backend_cfg[0]}" == "hip" ] ; then
        result+=" -Dalpaka_ACC_GPU_HIP_ENABLE=ON -Dalpaka_ACC_GPU_HIP_ONLY_MODE=ON"
        if [ $num_options -eq 2 ] ; then
            result+=" -DGPU_TARGETS=\"${backend_cfg[1]}\""
        else
            # If no architecture is given build for Radeon VII or MI50/60.
            result+=" -DGPU_TARGETS=gfx906"
        fi
    else
        echo "unsupported backend given '$1'" >&2
        exit 1
    fi

    echo "$result"
    exit 0
}
