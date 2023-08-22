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
mkdir build
cd build
cmake .. -Dcupla_BUILD_EXAMPLES=${CUPLA_BUILD_EXAMPLE} -Dcupla_ALPAKA_PROVIDER="external" -DBOOST_ROOT=/opt/boost/${CUPLA_BOOST_VERSION} -Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON
cmake --build .
cmake --install .
cd ${CUPLA_ROOT}
