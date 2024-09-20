#!/bin/bash

#
# Copyright 2023 Benjamin Worpitz, Jeffrey Kelling, Bernhard Manfred Gruber, Jan Stephan
# SPDX-License-Identifier: MPL-2.0
#

set +xv
source ./script/setup_utilities.sh

echo_green "<SCRIPT: run_generate>"

: "${CMAKE_CXX_COMPILER?'CMAKE_CXX_COMPILER must be specified'}"
: "${alpaka_CXX_STANDARD?'alpaka_CXX_STANDARD must be specified'}"

#-------------------------------------------------------------------------------

# create a cmake variable definition if an environment variable exists
#
# This function can not handle environment variables with spaces in its content.
#
# @param $1 cmake/environment variable name
#
# @result if $1 exists cmake variable definition else nothing is returned
#
# @code{.bash}
# FOO=ON
# echo "$(env2cmake FOO)" # returns "-DFOO=ON"
# echo "$(env2cmake BAR)" # returns nothing
# @endcode
function env2cmake()
{
    if [ ! -z "${!1}" ] ; then
        echo -n "-D$1=${!1}"
    fi
}

#-------------------------------------------------------------------------------
if [ ! -z "${CMAKE_CXX_FLAGS+x}" ]
then
    echo "CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
fi
if [ ! -z "${CMAKE_CXX_COMPILER+x}" ]
then
    echo "CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
fi
if [ ! -z "${CMAKE_EXE_LINKER_FLAGS+x}" ]
then
    echo "CMAKE_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}"
fi

ALPAKA_CI_CMAKE_EXECUTABLE=cmake
if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
then
    ALPAKA_CI_CMAKE_EXECUTABLE="${ALPAKA_CI_CMAKE_DIR}/bin/cmake"
fi

ALPAKA_CI_CMAKE_GENERATOR_PLATFORM=
if [ "$ALPAKA_CI_OS_NAME" = "Linux" ] || [ "$ALPAKA_CI_OS_NAME" = "macOS" ]
then
    ALPAKA_CI_CMAKE_GENERATOR="Unix Makefiles"
elif [ "$ALPAKA_CI_OS_NAME" = "Windows" ]
then
    : ${ALPAKA_CI_CL_VER?"ALPAKA_CI_CL_VER must be specified"}

    # Select the generator
    if [ "$ALPAKA_CI_CL_VER" = "2019" ]
    then
        ALPAKA_CI_CMAKE_GENERATOR="Visual Studio 16 2019"
    elif [ "$ALPAKA_CI_CL_VER" = "2022" ]
    then
        ALPAKA_CI_CMAKE_GENERATOR="Visual Studio 17 2022"
    fi
    ALPAKA_CI_CMAKE_GENERATOR_PLATFORM="-A x64"
fi

alpaka_CHECK_HEADERS=$ALPAKA_CI_ANALYSIS

mkdir -p build/
cd build/

"${ALPAKA_CI_CMAKE_EXECUTABLE}" --log-level=VERBOSE -G "${ALPAKA_CI_CMAKE_GENERATOR}" ${ALPAKA_CI_CMAKE_GENERATOR_PLATFORM}\
    -Dalpaka_BUILD_EXAMPLES=ON -DBUILD_TESTING=ON -Dalpaka_BUILD_BENCHMARKS=ON "$(env2cmake alpaka_ENABLE_WERROR)" \
    "$(env2cmake BOOST_ROOT)" -DBOOST_LIBRARYDIR="${ALPAKA_CI_BOOST_LIB_DIR}/lib" -DBoost_USE_STATIC_LIBS=ON -DBoost_USE_MULTITHREADED=ON -DBoost_USE_STATIC_RUNTIME=OFF -DBoost_ARCHITECTURE="-x64" \
    "$(env2cmake CMAKE_BUILD_TYPE)" "$(env2cmake CMAKE_CXX_FLAGS)" "$(env2cmake CMAKE_CXX_COMPILER)" "$(env2cmake CMAKE_EXE_LINKER_FLAGS)" "$(env2cmake CMAKE_CXX_EXTENSIONS)"\
    "$(env2cmake alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE)" "$(env2cmake alpaka_ACC_CPU_B_SEQ_T_THREADS_ENABLE)" \
    "$(env2cmake alpaka_ACC_CPU_B_TBB_T_SEQ_ENABLE)" \
    "$(env2cmake alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE)" "$(env2cmake alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE)" \
    "$(env2cmake TBB_DIR)" \
    "$(env2cmake alpaka_RELOCATABLE_DEVICE_CODE)" \
    "$(env2cmake alpaka_ACC_GPU_CUDA_ENABLE)" "$(env2cmake alpaka_ACC_GPU_CUDA_ONLY_MODE)" "$(env2cmake CMAKE_CUDA_ARCHITECTURES)" "$(env2cmake CMAKE_CUDA_COMPILER)" "$(env2cmake CMAKE_CUDA_FLAGS)" \
    "$(env2cmake alpaka_CUDA_FAST_MATH)" "$(env2cmake alpaka_CUDA_FTZ)" "$(env2cmake alpaka_CUDA_SHOW_REGISTER)" "$(env2cmake alpaka_CUDA_KEEP_FILES)" "$(env2cmake alpaka_CUDA_EXPT_EXTENDED_LAMBDA)" \
    "$(env2cmake alpaka_ACC_GPU_HIP_ENABLE)" "$(env2cmake alpaka_ACC_GPU_HIP_ONLY_MODE)" "$(env2cmake CMAKE_HIP_ARCHITECTURES)" "$(env2cmake CMAKE_HIP_COMPILER)" "$(env2cmake CMAKE_HIP_FLAGS)" \
    "$(env2cmake alpaka_ACC_SYCL_ENABLE)" "$(env2cmake alpaka_SYCL_ONEAPI_CPU)" "$(env2cmake alpaka_SYCL_ONEAPI_CPU_ISA)" \
    "$(env2cmake alpaka_DEBUG)" "$(env2cmake alpaka_CI)" "$(env2cmake alpaka_CHECK_HEADERS)" "$(env2cmake alpaka_CXX_STANDARD)" "$(env2cmake alpaka_USE_MDSPAN)" "$(env2cmake CMAKE_INSTALL_PREFIX)" \
    ".."

cd ..
