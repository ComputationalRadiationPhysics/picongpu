#!/bin/bash

# $1 - is the name of the subproject to test [pmacc, picongpu]

source $CI_PROJECT_DIR/share/ci/pmacc_env.sh

# compile header include consistency check
# use one build directory for all build configurations
cd $HOME
mkdir build_${1}_headerCI
cd build_${1}_headerCI
cmake $CMAKE_ARGS $code_DIR/test/${1}HeaderCheck
make -j $(nproc)
