#!/bin/bash

# This file is part of the PIConGPU.
# Copyright 2023 PIConGPU contributors
# Authors: Simeon Ehrig
# License: GPLv3+

# - the script installs a Python environment
# - generates a modified requirements.txt depending of the environment variables for pypicongpu
# - install the dependencies and runs the quick tests

export PICSRC=$CI_PROJECT_DIR
export PATH=$PATH:$PICSRC/bin
export PYTHONPATH=$PICSRC/lib/python:$PYTHONPATH
export PIC_EXAMPLES=$PICSRC/share/picongpu/examples

# use miniconda as python environment
apt update && apt install -y wget
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod u+x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda3
export PATH=/miniconda3/bin:$PATH
conda --version
source /miniconda3/etc/profile.d/conda.sh

# generates modified requirements.txt
conda create -n pypicongpu python=${PYTHON_VERSION}
conda activate pypicongpu
python3 --version
MODIFIED_REQUIREMENT_TXT=$CI_PROJECT_DIR/lib/python/picongpu/modified_requirements.txt
python3 $CI_PROJECT_DIR/share/ci/install/requirements_txt_modifier.py $CI_PROJECT_DIR/lib/python/picongpu/requirements.txt $MODIFIED_REQUIREMENT_TXT

echo "modified_requirements.txt: "
cat $MODIFIED_REQUIREMENT_TXT
echo ""

# run quick tests
pip3 install -r $MODIFIED_REQUIREMENT_TXT
cd $CI_PROJECT_DIR/test/python/picongpu
python3 -m quick
