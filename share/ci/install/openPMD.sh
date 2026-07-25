#!/bin/bash

set -e
set -o pipefail

# ADIOS2 is a dependency of  the pre compiled openPMD
export ADIOS2_ROOT="$(agc-manager -b adios2@2.10.2)"
export CMAKE_PREFIX_PATH=ADIOS2_ROOT:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=ADIOS2_ROOT/lib:$LD_LIBRARY_PATH
export PATH=ADIOS2_ROOT/bin:$PATH

# HDF5 is a dependency of  the pre compiled openPMD
export HDF5_ROOT="$(agc-manager -b hdf5@1.14.5)"
export CMAKE_PREFIX_PATH=HDF5_ROOT:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=HDF5_ROOT/lib:$LD_LIBRARY_PATH

if [ -z "$DISABLE_OpenPMD" ] ; then
    cd $CI_PROJECT_DIR

    export openPMD_VERSION=0.17.0

    if ! agc-manager -e openPMD-api@${openPMD_VERSION}; then
        export OPENPMD_ROOT=/opt/openPMD-api/$openPMD_VERSION

        export CMAKE_PREFIX_PATH=$GLM_ROOT:$CMAKE_PREFIX_PATH
        git clone https://github.com/openPMD/openPMD-api.git --depth 1 --branch $openPMD_VERSION
        mkdir buildOpenPMD
        cd buildOpenPMD
        cmake ../openPMD-api -DCMAKE_CXX_STANDARD=20 -DopenPMD_BUILD_CLI_TOOLS=OFF -DCMAKE_INSTALL_PREFIX=$OPENPMD_ROOT -DopenPMD_BUILD_EXAMPLES=OFF -DopenPMD_BUILD_TESTING=OFF -DopenPMD_USE_PYTHON=OFF -DopenPMD_USE_ADIOS2=ON
        make install
        cd ..

        export CMAKE_PREFIX_PATH=$OPENPMD_ROOT:$CMAKE_PREFIX_PATH
        export LD_LIBRARY_PATH=$OPENPMD_ROOT/lib:$LD_LIBRARY_PATH
    else
        export OPENPMD_ROOT="$(agc-manager -b openPMD-api@${openPMD_VERSION})"
        export CMAKE_PREFIX_PATH=$OPENPMD_ROOT:$CMAKE_PREFIX_PATH
        export LD_LIBRARY_PATH=$OPENPMD_ROOT/lib:$LD_LIBRARY_PATH
    fi
fi
