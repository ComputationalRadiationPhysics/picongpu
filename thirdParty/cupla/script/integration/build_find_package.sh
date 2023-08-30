#!/bin/bash

CUPLA_ROOT=$(pwd)

##########################
# update environment
##########################
if [ agc-manager -e cmake@3.22 -ne 0 ] ; then
    echo "CMake 3.22 is not available" >&2
    exit 1
else
    export PATH=$(agc-manager -b cmake@3.22)/bin:$PATH
fi

##########################
# create external project
##########################
mkdir /tmp/external_project
cd /tmp/external_project
# copy source file and CMakeLists.txt to external project to simulate external project
EXT_PROJECT_ROOT=$(pwd)
cp ${CUPLA_ROOT}/example/CUDASamples/cuplaVectorAdd/src/vectorAdd.cpp ${EXT_PROJECT_ROOT}
cp ${CUPLA_ROOT}/test/integration/cupla_find_package.cmake ${EXT_PROJECT_ROOT}/CMakeLists.txt
cd ${EXT_PROJECT_ROOT}

##########################
# build external project
##########################
mkdir build install
export LD_LIBRARY_PATH=${EXT_PROJECT_ROOT}/install/lib:${LD_LIBRARY_PATH}
echo $LD_LIBRARY_PATH
cd build
cmake .. -Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON -DBOOST_ROOT=/opt/boost/${CUPLA_BOOST_VERSION} -DCMAKE_INSTALL_PREFIX=../install -DBUILD_SHARED_LIBS=ON
cmake --build .
cmake --install .

##########################
# test build
##########################
cd ../install
ls
bin/cuplaVectorAdd
