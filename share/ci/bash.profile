#!/bin/bash

# setup dependencies for PIConGPU for CMake and runtime usage

set -e
set -o pipefail

if [ -d "/opt/pngwriter" ] ; then
  export PNGWRITER_ROOT=/opt/pngwriter/0.7.0
else
  # pngwriter is currently install to the / instead of /opt
  export PNGWRITER_ROOT=/pngwriter/0.7.0
fi
export CMAKE_PREFIX_PATH=$PNGWRITER_ROOT:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$PNGWRITER_ROOT/lib:$LD_LIBRARY_PATH

# set environment variable for path to tpls for PyPIConGPU runner
export PIC_SYSTEM_TEMPLATE_PATH=${PIC_SYSTEM_TEMPLATE_PATH:-"etc/picongpu/bash"}

if [ -z "$DISABLE_ISAAC" ] ; then
  export ICET_ROOT=/opt/icet/2.9.0
  export CMAKE_PREFIX_PATH=$ICET_ROOT/lib:$CMAKE_PREFIX_PATH
  export LD_LIBRARY_PATH=$ICET_ROOT/lib:$LD_LIBRARY_PATH

  export ISAAC_ROOT=/opt/isaac/1.6.0-dev-custom
  # install cusom version of isaac
  source $CI_PROJECT_DIR/share/ci/install/isaac.sh

  export CMAKE_PREFIX_PATH=$ISAAC_ROOT:$CMAKE_PREFIX_PATH
  export LD_LIBRARY_PATH=$ISAAC_ROOT/lib:$LD_LIBRARY_PATH
fi
