#!/bin/bash

# This file is part of PIConGPU.
# Copyright 2023-2024 PIConGPU contributors
# Authors: Simeon Ehrig
# License: GPLv3+

# - the script installs a Python environment
# - generates a modified requirements.txt depending of the environment variables for pypicongpu
# - install the dependencies and runs the pytest tests

set -e
set -o pipefail

function script_error {
    echo -e "\e[31mERROR: ${1}\e[0m"
    exit 1
}

export PICSRC=$CI_PROJECT_DIR
export PATH=$PATH:$PICSRC/bin
echo 'preset = "bash"' >/.picongpurc.toml

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
# install requirements of pyproject_toml_modifier.txt
pip3 install -r ${CI_PROJECT_DIR}/share/ci/install/pyproject_toml_modifier_requirements.txt
PYPROJECT_TOML_PATH=${CI_PROJECT_DIR}/lib/python/pyproject.toml

python3 $CI_PROJECT_DIR/share/ci/install/pyproject_toml_modifier.py \
    -i $PYPROJECT_TOML_PATH \
    -o $PYPROJECT_TOML_PATH \

# uninstall requirements of pyproject_toml_modifier.txt
pip3 uninstall -y -r ${CI_PROJECT_DIR}/share/ci/install/pyproject_toml_modifier_requirements.txt

echo "modified pyproject.toml: "
cat $PYPROJECT_TOML_PATH
echo ""

# install pypicongpu dependencies (including pytest for testing)
pip3 install -e "${CI_PROJECT_DIR}/lib/python/[test]"

# run quick tests
cd $CI_PROJECT_DIR/lib/python/test/picongpu
python3 -m pytest quick/

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

    # set C++ compiler
    export CXX=$CXX_VERSION
    # execute the compiling test
    python3 -m pytest compiling/ -v
fi

# running the end-to-end (actual simulation) tests is optional
# for the end-to-end test we need: cmake, boost and openmpi
# openmpi is available without extra work
if [ ! -z ${PYTHON_END_TO_END_TEST+x} ]; then
    export PIC_BACKEND=serial
    # the CI job runs as root, but OpenMPI refuses to start as root,
    # so the simulation would be aborted by mpiexec before producing output
    export OMPI_ALLOW_RUN_AS_ROOT=1
    export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
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

    # set C++ compiler
    export CXX=$CXX_VERSION
    # Deselect the free-formula-density module for now: its simulation is
    # compile-time too heavy for the CI runner -- a single translation unit
    # (plugins/binning/BinningDispatcher) peaks at ~15 GiB and gcc gets
    # OOM-killed regardless of build parallelism. Re-enable once that
    # translation unit is lightened or the runner provides more memory.
    python3 -m pytest end_to_end/ -v --ignore=end_to_end/test_free_formula_density.py
fi
