#!/bin/bash

source $CI_PROJECT_DIR/share/ci/pmacc_env.sh

# compile header include consistency check
# use one build directory for all build configurations
cd $HOME
mkdir buildPMaccHeaderCI
cd buildPMaccHeaderCI
cmake $CMAKE_ARGS $code_DIR/test/pmaccHeaderCheck
make -j $(nproc)
