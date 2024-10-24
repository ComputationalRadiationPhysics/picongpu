#!/bin/bash

set -e
set -o pipefail

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

if [[ "$PIC_BACKEND" =~ hip.* ]] ; then
    if [ -z "$CI_GPU_ARCH" ] ; then
        # In case the CI runner is not providing a GPU architecture e.g. a CPU runner set the architecture
        # to Radeon VII or MI50/60.
        export GPU_TARGETS="gfx906"
    fi
    export PIC_CMAKE_ARGS="$PIC_CMAKE_ARGS -DCMAKE_HIP_ARCHITECTURES=$GPU_TARGETS"
fi

###################################################
# cmake config builder
###################################################

PIC_CONST_ARGS=""
# to save compile time reduce the isaac functor chain length to one
PIC_CONST_ARGS="${PIC_CONST_ARGS} -DPIC_CI_COMPILE=ON -DISAAC_MAX_FUNCTORS=1 -DCMAKE_BUILD_TYPE=${PIC_BUILD_TYPE}"
CMAKE_ARGS="${PIC_CONST_ARGS} ${PIC_CMAKE_ARGS} -DCMAKE_CXX_COMPILER=${CXX_VERSION} -DBOOST_ROOT=/opt/boost/${BOOST_VERSION}"

# check and activate if clang should be used as CUDA device compiler
if [ -n "$CI_CLANG_AS_CUDA_COMPILER" ] ; then
  export PATH="$(agc-manager -b cuda)/bin:$PATH"
  CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_COMPILER=${CXX_VERSION}"
fi

if [[ "$PIC_TEST_CASE_FOLDER" =~ .*Empty.* ]] ; then
    # For the empty test case (default param files) we disable all optional dependencies to have at least one check
    # where all dependencies are disabled.
    CMAKE_ARGS="$CMAKE_ARGS -DPIC_USE_ISAAC=OFF -DPIC_USE_openPMD=OFF -DPIC_USE_PNGwriter=OFF -DPIC_USE_FFTW3=OFF"
else
    # enforce optional dependencies
    CMAKE_ARGS="$CMAKE_ARGS -DPIC_USE_openPMD=ON -DPIC_USE_PNGwriter=ON -DPIC_USE_FFTW3=ON"

    # ISAAC together with the example FoilLCT is to complex therefore the CI is always running out of memory.
    # ISAAC is disabled until someone adds support for alpaka 1.2.0
    # re-enable tho following code if somene fixes the ISAAC issues.
    CMAKE_ARGS="$CMAKE_ARGS -DPIC_USE_ISAAC=OFF"
    if [[ "$PIC_TEST_CASE_FOLDER" =~ .*FoilLCT.* ]] ; then
         export CI_CPUS=1
    #elif [ -z "$DISABLE_ISAAC" ] ; then
    #    CMAKE_ARGS="$CMAKE_ARGS -DPIC_USE_ISAAC=ON"
    fi
fi

# Test is running out of memory, therefore we do not run in parallel.
if [[ "$PIC_TEST_CASE_FOLDER" =~ .*CollisionsBeamRelaxation.* ]] ; then
  export CI_CPUS=1
fi

# workaround for clang cuda
# HDF5 from the apt sources is pulling -D_FORTIFY_SOURCE=2 into the compile flags
# this workaround is creating a warning about the double definition of _FORTIFY_SOURCE
#
# Workaround will be removed after the test container are shipped with a self compiled HDF5
if [[ $CXX_VERSION =~ ^clang && $PIC_BACKEND =~ ^cuda ]] ; then
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CXX_FLAGS=-D_FORTIFY_SOURCE=0"
fi

###################################################
# build an run tests
###################################################

# use one build directory for all build configurations
cd $HOME
mkdir buildCI
cd buildCI

export picongpu_DIR=$CI_PROJECT_DIR
export PATH=$picongpu_DIR/bin:$PATH

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

if [ "$PIC_TEST_CASE_FOLDER" == "examples/" ] || [ "$PIC_TEST_CASE_FOLDER" == "tests/" ] ||  [ "$PIC_TEST_CASE_FOLDER" == "benchmarks/" ] ; then
    extended_compile_options="-l"
fi

# test compiling
error_code=$(pic-compile -q -c"$CMAKE_ARGS" $extended_compile_options -j $PIC_PARALLEL_BUILDS ${picongpu_DIR}/share/picongpu/$PIC_TEST_CASE_FOLDER  . 2>&1 > pic_compile.log && echo "0" || echo "1")
cat pic_compile.log
for test_case in $(ls -w1 ./build) ; do
    if [ -f  "build/$test_case/returnCode" ] ; then
        returnCode=$(cat "build/$test_case/returnCode")
        if [ "$returnCode" != "0" ] ; then
            echo -e "\033[0;31m compile FAILED - $test_case \033[0m"
            cat "build/$test_case/compile.log"
        else
            echo -e "\033[0;32m compile PASSED - $test_case \033[0m"
        fi
    else
        echo -e "\033[0;33m compile NOT tested - $test_case \033[0m"
    fi
done
if [ "$error_code" != "0" ] ; then
    exit 1
fi
# runtime test (call --help)
for test_case_folder in $(ls params/*/* -d -w1) ; do
    export LD_LIBRARY_PATH=/opt/boost/${BOOST_VERSION}/lib:$LD_LIBRARY_PATH
    echo -e "\033[0;33m runtime test- $(basename $test_case_folder) \033[0m"
    ${test_case_folder}/bin/picongpu --help
done
