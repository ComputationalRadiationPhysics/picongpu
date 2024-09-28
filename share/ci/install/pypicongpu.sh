#!/bin/bash

# This file is part of PIConGPU.
# Copyright 2023-2023 PIConGPU contributors
# Authors: Simeon Ehrig
# License: GPLv3+

# - the script installs a Python environment
# - generates a modified requirements.txt depending of the environment variables for pypicongpu
# - install the dependencies and runs the quick tests

set -e
set -o pipefail

function script_error {
    echo -e "\e[31mERROR: ${1}\e[0m"
    exit 1
}

export PICSRC=$CI_PROJECT_DIR
export PATH=$PATH:$PICSRC/bin
export PYTHONPATH=$PICSRC/lib/python:$PYTHONPATH
export PIC_EXAMPLES=$PICSRC/share/picongpu/examples

cd $CI_PROJECT_DIR

# use miniconda as python environment
apt update && apt install -y curl
cd /tmp/
curl -Ls https://github.com/mamba-org/micromamba-releases/releases/download/1.5.9-0/micromamba-linux-64.tar.bz2 | tar -xvj bin/micromamba
export MAMBA_ROOT_PREFIX=/tmp/mamba-forge/
mkdir -p "${MAMBA_ROOT_PREFIX}"
eval "$(./bin/micromamba shell hook -s posix)"
export PATH=$(pwd -P)/bin:$PATH
micromamba --version
micromamba config append channels conda-forge
micromamba config set channel_priority strict

cd $CI_PROJECT_DIR
# generates modified requirements.txt
micromamba create -n pypicongpu python=${PYTHON_VERSION} --ssl-verify false
micromamba activate pypicongpu
python3 --version
MODIFIED_REQUIREMENT_TXT_PICMI=$CI_PROJECT_DIR/lib/python/picongpu/picmi/modified_requirements.txt
MODIFIED_REQUIREMENT_TXT_PYPICONGPU=$CI_PROJECT_DIR/lib/python/picongpu/pypicongpu/modified_requirements.txt
python3 $CI_PROJECT_DIR/share/ci/install/requirements_txt_modifier.py $CI_PROJECT_DIR/lib/python/picongpu/picmi/requirements.txt $MODIFIED_REQUIREMENT_TXT_PICMI
python3 $CI_PROJECT_DIR/share/ci/install/requirements_txt_modifier.py $CI_PROJECT_DIR/lib/python/picongpu/pypicongpu/requirements.txt $MODIFIED_REQUIREMENT_TXT_PYPICONGPU

echo "modified_requirements.txt: "
cat $MODIFIED_REQUIREMENT_TXT_PICMI
cat $MODIFIED_REQUIREMENT_TXT_PYPICONGPU
echo ""

# run quick tests
pip3 install -r $MODIFIED_REQUIREMENT_TXT_PICMI
pip3 install -r $MODIFIED_REQUIREMENT_TXT_PYPICONGPU
cd $CI_PROJECT_DIR/test/python/picongpu
python3 -m quick

# executing the compiling tests is optional
# for the compiling test we need: cmake, boost and openmpi
# openmpi is available without extra work
if [ ! -z ${PYTHON_COMPILING_TEST+x} ]; then
    export PIC_BACKEND=omp2b
    # setup cmake
    if [ ! -z ${CMAKE_VERSION+x} ]; then
        if agc-manager -e cmake@${CMAKE_VERSION} ; then
            export PATH=$(agc-manager -b cmake@${CMAKE_VERSION})/bin:$PATH
        else
            script_error "No implementation to install cmake ${CMAKE_VERSION}"
        fi
    else
        script_error "CMAKE_VERSION is not defined"
    fi

    # setup boost
    if [ ! -z ${BOOST_VERSION+x} ]; then
        if agc-manager -e boost@${BOOST_VERSION} ; then
            export CMAKE_PREFIX_PATH=$(agc-manager -b boost@${BOOST_VERSION}):$CMAKE_PREFIX_PATH
        else
            script_error "No implementation to install boost ${BOOST_VERSION}"
        fi
    else
        script_error "BOOST_VERSION is not defined"
    fi

    # execute the compiling test
    python3 -m compiling -v
fi
