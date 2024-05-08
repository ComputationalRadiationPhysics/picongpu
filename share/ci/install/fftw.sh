#!/bin/bash

set -e
set -o pipefail

cd $CI_PROJECT_DIR

echo "load/install fftw3"

if ! agc-manager -e fftw3; then
    apt install -y libfftw3-mpi-dev libfftw3-dev pkg-config
else
    FFTW3_BASE_PATH="$(agc-manager -b fftw3)"
    export CMAKE_PREFIX_PATH=$FFTW3_BASE_PATH:$CMAKE_PREFIX_PATH
fi
