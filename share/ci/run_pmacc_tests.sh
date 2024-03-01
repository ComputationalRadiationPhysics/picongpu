#!/bin/bash

source $CI_PROJECT_DIR/share/ci/pmacc_env.sh

# compile and run catch2 tests
# use one build directory for all build configurations
cd $HOME
mkdir buildPMaccCI
cd buildPMaccCI
cmake $CMAKE_ARGS $code_DIR/include/pmacc
make

ctest -V

# With HIP we run into MPI linker issues, therefore GoL tests are disabled for now
# clang+cuda is running into compiler ptx errors when code for sm_60 is generated.
# @todo analyse and fix MPI linker issues
if ! [[ "$PIC_BACKEND" =~ hip.* || ($CXX_VERSION =~ ^clang && $PIC_BACKEND =~ ^cuda) ]] ; then
  ## compile and test game of life
  export GoL_folder=$HOME/buildGoL
  mkdir -p $GoL_folder
  cd $GoL_folder
  # release compile to avoid issues with cuda <11.7 https://github.com/alpaka-group/alpaka/issues/2035
  cmake $CMAKE_ARGS -DGOL_RELEASE=ON $code_DIR/share/pmacc/examples/gameOfLife2D
  make -j $PMACC_PARALLEL_BUILDS
  # execute on one device
  ./gameOfLife -d 1 1 -g 64 64  -s 100 -p 1 1

  ## compile and test heat equation
  export heatEq_folder=$HOME/heatEq
  mkdir -p $heatEq_folder
  cd $heatEq_folder
  # release compile to avoid issues with cuda <11.7 https://github.com/alpaka-group/alpaka/issues/2035
  cmake $CMAKE_ARGS -DHEATEQ_RELEASE=ON $code_DIR/share/pmacc/examples/heatEquation2D
  make -j $PMACC_PARALLEL_BUILDS
  # execute with  = 4
  if [ "$USE_MPI_AS_ROOT_USER" == "ON" ]; then
    mpirun --allow-run-as-root -npernode 4 -n 4 ./heatEq
  fi
fi
