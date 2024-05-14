#!/bin/bash

set -e
set -o pipefail

if [ -z "$DISABLE_ISAAC" ] ; then
    cd $CI_PROJECT_DIR

    export GLM_VERSION=1.0.1
    export GLM_ROOT=/opt/glm/$GLM_VERSION
    export CMAKE_PREFIX_PATH=$GLM_ROOT:$CMAKE_PREFIX_PATH
    git clone https://github.com/g-truc/glm.git --depth 1 --branch $GLM_VERSION
    cd glm
    mkdir build
    cd build
    cmake ../ -DCMAKE_INSTALL_PREFIX=$GLM_ROOT -DGLM_TEST_ENABLE=OFF
    make install

    cd $CI_PROJECT_DIR
    git clone https://github.com/ComputationalRadiationPhysics/isaac.git
    cd isaac
    git checkout 6bf8e189142da7ab16f8367316fb21dcaa7e3b78
    mkdir build_isaac
    cd build_isaac
    cmake ../lib/ -DCMAKE_INSTALL_PREFIX=$ISAAC_ROOT
    make install
    cd ..
fi
