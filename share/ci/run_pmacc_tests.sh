#!/bin/bash

set -e
set -o pipefail

source $CI_PROJECT_DIR/share/ci/backendFlags.sh

# the default build type is Release
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
# build an run tests
###################################################

export code_DIR=$CI_PROJECT_DIR

PMACC_PARALLEL_BUILDS=$(nproc)
# limit to $CI_MAX_PARALLELISM parallel builds to avoid out of memory errors
# CI_MAX_PARALLELISM is a configured variable in the CI web interface
if [ $PMACC_PARALLEL_BUILDS -gt $CI_MAX_PARALLELISM ] ; then
    PMACC_PARALLEL_BUILDS=$CI_MAX_PARALLELISM
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

# With HIP we run into MPI linker issues, therefore GoL tests are disabled for now
# clang+cuda is running into compiler ptx errors when code for sm_60 is generated.
# @todo analyse and fix MPI linker issues
if ! [[ "$PIC_BACKEND" =~ hip.* || ($CXX_VERSION =~ ^clang && $PIC_BACKEND =~ ^cuda) ]] ; then
  ## compile and test game of life
  export GoL_folder=$HOME/buildGoL
  mkdir -p $GoL_folder
  cd $GoL_folder
  # release compile to avoid issues with cuda <11.7 https://github.com/alpaka-group/alpaka/issues/2035
  cmake $CMAKE_ARGS -DGOL_RELEASE=ON $code_DIR/share/pmacc/examples/gameOfLife2D
  make -j $PMACC_PARALLEL_BUILDS
  # execute on one device
  ./gameOfLife -d 1 1 -g 64 64  -s 100 -p 1 1
fi
