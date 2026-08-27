#/usr/bin/env bash

# set default values for unset environment variables

export CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS:=""}
export CMAKE_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS:=""}
export CMAKE_CXX_EXTENSIONS=${CMAKE_CXX_EXTENSIONS:="ON"}
export CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX:=""}

export CMAKE_CUDA_FLAGS=${CMAKE_CUDA_FLAGS:=""}
export CMAKE_HIP_FLAGS=${CMAKE_HIP_FLAGS:=""}
export ALPAKA_CI_TBB_DIR=${ALPAKA_CI_TBB_DIR:=""}

# increase the maximum stack size to work around a limitation of ROCm (see https://github.com/ROCm/ROCm/issues/4751)

ulimit -s hard
