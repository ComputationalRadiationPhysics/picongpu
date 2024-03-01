#!/bin/bash

set -e
set -o pipefail

if [ -z "$DISABLE_ISAAC" ] ; then
    cd $CI_PROJECT_DIR

    export GLM_ROOT=/opt/glm/0.9.9.9-dev
    export CMAKE_PREFIX_PATH=$GLM_ROOT:$CMAKE_PREFIX_PATH
    git clone https://github.com/g-truc/glm.git
    cd glm
    git checkout 6ad79aae3eb5bf809c30bf1168171e9e55857e45
    mkdir build
    cd build
    cmake ../ -DCMAKE_INSTALL_PREFIX=$GLM_ROOT -DGLM_TEST_ENABLE=OFF
    make install

    cd $CI_PROJECT_DIR
    git clone https://github.com/ComputationalRadiationPhysics/isaac.git
    cd isaac
    git checkout 195420ca1c43f6148c7f6c3a10ad9cad32e04d6a
    mkdir build_isaac
    cd build_isaac
    cmake ../lib/ -DCMAKE_INSTALL_PREFIX=$ISAAC_ROOT
    make install
    cd ..
fi
