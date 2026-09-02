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

    export JANSSON_VERSION=v2.14.1
    export JANSSON_ROOT=/opt/jansson/$JANSSON_VERSION
    export CMAKE_PREFIX_PATH=$JANSSON_ROOT:$CMAKE_PREFIX_PATH
    export LD_LIBRARY_PATH=$JANSSON_ROOT/lib:$LD_LIBRARY_PATH
    git clone https://github.com/akheron/jansson.git --depth 1 \
        --branch $JANSSON_VERSION
    cd $ISAAC_SOURCE_DIR/jansson
    mkdir build
    cd build
    cmake ../ -DCMAKE_INSTALL_PREFIX=$JANSSON_ROOT
    make install

    cd $CI_PROJECT_DIR
    git clone https://github.com/ComputationalRadiationPhysics/isaac.git
    cd isaac
    git checkout bb554c7ea7e62712309946f971a905e3835da6f9
    mkdir build_isaac
    cd build_isaac
    cmake ../lib/ -DCMAKE_INSTALL_PREFIX=$ISAAC_ROOT
    make install
    cd ..
fi
