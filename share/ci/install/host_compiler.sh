#!/bin/bash

set -e
set -o pipefail

# install the compiler selected via CXX_VERSION and export it as C++ and C
# compiler. This is used by jobs which select the device compiler independently
# of the host compiler, e.g. nvcc with g++ or clang++ as host compiler.
if [[ "${CXX_VERSION}" =~ ^clang\+\+ ]] ; then
    source $CI_PROJECT_DIR/share/ci/install/clang.sh
else
    source $CI_PROJECT_DIR/share/ci/install/gcc.sh
fi

# tell CMake which host compiler to use for CUDA
# CMake reads CUDAHOSTCXX during its CUDA compiler detection and uses it as
# CMAKE_CUDA_HOST_COMPILER (passed to nvcc as -ccbin), which takes precedence
# over the CMAKE_CXX_COMPILER fallback in include/pmacc/PMaccConfig.cmake
if ! CUDAHOSTCXX="$(command -v ${CXX_VERSION})" ; then
    echo "ERROR: host compiler '${CXX_VERSION}' was not found" >&2
    exit 1
fi
export CUDAHOSTCXX
echo "CUDA host compiler -> ${CUDAHOSTCXX}"
"${CUDAHOSTCXX}" --version | head -n 1
