#!/bin/bash
# Execute PIConGPU's unit tests

set -e
set -o pipefail

export code_DIR=$CI_PROJECT_DIR
source $code_DIR/share/ci/backendFlags.sh

# the default build type is Release
# to change the build type, you must set the environment variable PIC_BUILD_TYPE
if [[ ! -v PIC_BUILD_TYPE ]] ; then
    PIC_BUILD_TYPE=Release ;
fi

if [[ "$CI_RUNNER_TAGS" =~ .*cpuonly.* ]] ; then
    # In cases where the compile-only job is executed on a GPU runner but with different kinds of accelerators
    # we need to reset the variables to avoid compiling for the wrong architecture and accelerator.
    unset CI_GPUS
    unset CI_GPU_ARCH
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

###################################################
# cmake config builder
###################################################

PIC_CONST_ARGS=""
PIC_CONST_ARGS="${PIC_CONST_ARGS} -DCMAKE_BUILD_TYPE=${PIC_BUILD_TYPE}"
CMAKE_ARGS="${PIC_CONST_ARGS} ${PIC_CMAKE_ARGS} -DCMAKE_CXX_COMPILER=${CXX_VERSION} -DBOOST_ROOT=/opt/boost/${BOOST_VERSION}"
CMAKE_ARGS="$CMAKE_ARGS -DUSE_MPI_AS_ROOT_USER=ON"

# check and activate if clang should be used as CUDA device compiler
if [ -n "$CI_CLANG_AS_CUDA_COMPILER" ] ; then
  export PATH="$(agc-manager -b cuda)/bin:$PATH"
  CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_COMPILER=${CXX_VERSION}"
fi

alpaka_backend=$(get_backend_flags ${PIC_BACKEND})
CMAKE_ARGS="$CMAKE_ARGS $alpaka_backend"

###################################################
# build and run unit tests
###################################################

# adjust number of parallel builds to avoid out of memory errors
# PIC_BUILD_REQUIRED_MEM_BYTES is a configured variable in the CI web interface
PIC_PARALLEL_BUILDS=$(($CI_RAM_BYTES_TOTAL/$PIC_BUILD_REQUIRED_MEM_BYTES))

# limit to number of available cores
if [ $PIC_PARALLEL_BUILDS -gt $CI_CPUS ] ; then
    PIC_PARALLEL_BUILDS=$CI_CPUS
fi

# CI_MAX_PARALLELISM is a configured variable in the CI web interface
if [ $PIC_PARALLEL_BUILDS -gt $CI_MAX_PARALLELISM ] ; then
    PIC_PARALLEL_BUILDS=$CI_MAX_PARALLELISM
fi

## run unit tests
export unitTest_folder=$HOME/buildPICUnitTest
mkdir -p $unitTest_folder
cd $unitTest_folder

echo -e "\033[0;32m///////////////////////////////////////////////////"
echo "PIC_BUILD_REQUIRED_MEM_BYTES-> ${PIC_BUILD_REQUIRED_MEM_BYTES}"
echo "CI_RAM_BYTES_TOTAL          -> ${CI_RAM_BYTES_TOTAL}"
echo "CI_CPUS                     -> ${CI_CPUS}"
echo "CI_MAX_PARALLELISM          -> ${CI_MAX_PARALLELISM}"
echo "number of processor threads -> $(nproc)"
echo "number of parallel builds   -> $PIC_PARALLEL_BUILDS"
echo "cmake version               -> $(cmake --version | head -n 1)"
echo "build directory             -> $(pwd)"
echo "CMAKE_ARGS                  -> ${CMAKE_ARGS}"
echo "accelerator                 -> ${PIC_BACKEND}"
echo "input set                   -> ${PIC_TEST_CASE_FOLDER}"
echo -e "/////////////////////////////////////////////////// \033[0m \n\n"

cmake $CMAKE_ARGS $code_DIR/share/picongpu/unit
make -j $PIC_PARALLEL_BUILDS
# execute on one device
ctest -V
