#!/bin/bash

set -e
set -o pipefail

# the default build type is Release
# if neccesary, you can rerun the pipeline with another build type-> https://docs.gitlab.com/ee/ci/pipelines.html#manually-executing-pipelines
# to change the build type, you must set the environment variable PMACC_BUILD_TYPE
if [[ ! -v PMACC_BUILD_TYPE ]] ; then
    PMACC_BUILD_TYPE=Release;
fi

###################################################
# cmake config builder
###################################################

PMACC_CONST_ARGS=""
# to save compile time reduce the isaac functor chain length to one
PMACC_CONST_ARGS="${PMACC_CONST_ARGS} -DCMAKE_BUILD_TYPE=${PMACC_BUILD_TYPE}"
CMAKE_ARGS="${PMACC_CONST_ARGS} ${PIC_CMAKE_ARGS} -DCMAKE_CXX_COMPILER=${CXX_VERSION} -DBOOST_ROOT=/opt/boost/${BOOST_VERSION}"
# allow root user to execute MPI
CMAKE_ARGS="$CMAKE_ARGS -DUSE_MPI_AS_ROOT_USER=ON"


# check and activate if clang should be used as CUDA device compiler
if [ -n "$CI_CLANG_AS_CUDA_COMPILER" ] ; then
    export PATH="$(agc-manager -b cuda)/bin:$PATH"
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_COMPILER=${CXX_VERSION}"
fi

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
            result+=" -DCMAKE_CUDA_ARCHITECTURES=35"
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

###################################################
# build an run tests
###################################################

export code_DIR=$CI_PROJECT_DIR

PMACC_PARALLEL_BUILDS=$(nproc)
# limit to $CI_MAX_PARALLELISM parallel builds to avoid out of memory errors
# CI_MAX_PARALLELISM is a configured variable in the CI web interface
if [ $PMACC_PARALLEL_BUILDS -gt $CI_MAX_PARALLELISM ] ; then
    PMACC_PARALLEL_BUILDS=$CI_MAX_PARALLELISM
fi

if [ -n "$CI_GPUS" ] ; then
    # select randomly a device if multiple exists
    # CI_GPUS is provided by the gitlab CI runner
    SELECTED_DEVICE_ID=$((RANDOM%CI_GPUS))
    export HIP_VISIBLE_DEVICES=$SELECTED_DEVICE_ID
    export CUDA_VISIBLE_DEVICES=$SELECTED_DEVICE_ID
    echo "selected device '$SELECTED_DEVICE_ID' of '$CI_GPUS'"
else
    echo "No GPU device selected because environment variable CI_GPUS is not set."
fi

if [[ "$PIC_BACKEND" =~ hip.* ]] || [[ "$PIC_BACKEND" =~ cuda.* ]] ; then
    if [ -n "$CI_GPU_ARCH" ] ; then
        export PIC_BACKEND="${PIC_BACKEND}:${CI_GPU_ARCH}"
    fi
fi

alpaka_backend=$(get_backend_flags ${PIC_BACKEND})
CMAKE_ARGS="$CMAKE_ARGS $alpaka_backend"

echo -e "\033[0;32m///////////////////////////////////////////////////"
echo "number of processor threads -> $(nproc)"
echo "number of parallel builds -> $PMACC_PARALLEL_BUILDS"
echo "cmake version   -> $(cmake --version | head -n 1)"
echo "build directory -> $(pwd)"
echo "CMAKE_ARGS      -> ${CMAKE_ARGS}"
echo "accelerator     -> ${PIC_BACKEND}"
echo -e "/////////////////////////////////////////////////// \033[0m \n\n"

# disable warning if infiniband is not used
export OMPI_MCA_btl_base_warn_component_unused=0
export LD_LIBRARY_PATH=/opt/boost/${BOOST_VERSION}/lib:$LD_LIBRARY_PATH

# compile header include consistency check
# use one build directory for all build configurations
cd $HOME
mkdir buildPMaccHeaderCI
cd buildPMaccHeaderCI
cmake $CMAKE_ARGS $code_DIR/test/pmaccHeaderCheck
make -j $(nproc)

# compile and run catch2 tests
# use one build directory for all build configurations
cd $HOME
mkdir buildPMaccCI
cd buildPMaccCI
cmake $CMAKE_ARGS $code_DIR/include/pmacc
make

ctest -V
