#!/bin/bash

##########################
# update environment
##########################
if [ agc-manager -e cmake@3.22 -ne 0 ] ; then
    echo "CMake 3.22 is not available" >&2
    exit 1
else
    export PATH=$(agc-manager -b cmake@3.22)/bin:$PATH
fi

CUPLA_ROOT=$(pwd)
cd alpaka
mkdir build
cd build
cmake .. -DBOOST_ROOT=/opt/boost/${CUPLA_BOOST_VERSION}
cmake --build .
cmake --install .
cd ${CUPLA_ROOT}
# Remove alpaka repository so that it cannot be used accidentally via add_subdirectory()
rm -rf alpaka
