#!/bin/bash
#
# This file is part of PIConGPU.
# Copyright 2023 PIConGPU contributors
# Authors: Mika Soren Voss, Hannes Wolf, Klaus Steiniger
# License: GPLv3+
#

set -o pipefail

function absolute_path()
{
    builtin cd -- "$1" &> /dev/null && pwd
}

help()
{
  echo "Simulate a current-carrying wire in a small and large volume in order to quantify PML performance by"
  echo "comparing values of the electric field close to the wire between both simulations after many timesteps."
  echo ""
  echo "Usage:"
  echo "    (1) Change current working directory to direcectory where the include directory of the setup is located"
  echo "    (2) execute ci.sh from this directory"
  echo ""
  echo "Options"
  echo "-h | --help                   - show help"
  echo ""
}

## not used at the moment
currentPath=$(cd `dirname $0` && pwd)
currentPath=$(absolute_path $currentPath)


#####################
## option handling ##
#####################
# options may be followed by
# - one colon to indicate they have a required argument
OPTS=`getopt -o h -l help -- "$@"`
if [ $? != 0 ] ; then
    # something went wrong, getopt will put out an error message for us
    exit 1
fi

eval set -- "$OPTS"

# parser
while true ; do
    case "$1" in
        -h|--help)
            echo -e "$(help)"
            shift
            exit 0
            ;;
        --) shift; break;;
    esac
    shift
done


############################
## build and run picongpu ##
############################
if ! [ -d "./include" ] ; then
  echo "Execute ci.sh from the directory where the simulation include dir is located!"
  exit 1
fi

## create simulation data directory
simPath="./simOutput"

if [ -d "$simPath" ] ; then
  echo "Destination path already in use, cannot create new folder" >&2
  exit 1
fi

mkdir -p $simPath

# use absolut path's
simPath=$(absolute_path $simPath)

for i in {0..1}; do
  ## i=0: build and run the small volume 60x60x60 sim w/ absorbers
  ## i=1: build and run the large volume 1060x1060x1060 reference simulation
  pic-build -t $i
  ret_build=$?

  if [ $ret_build -eq 0 ] ; then
    cd $simPath
    echo "Run setup ${i}!"
    if [ $i -eq 0 ]; then
      # make sure that the grid size along z matches
      # the grid size along z of the reference simulation below
      mpiexec -n 1 ../bin/picongpu -d 1 1 1 -g 60 60 660 --periodic 0 0 1 -s 600 \
        --fields_energy.period 10 \
        --openPMD.period 100 --openPMD.ext bp --openPMD.file simData_test

    elif [ $i -eq 1 ]; then
      # match the grid size with the grid size defined in `flags[1]` of cmakeFlags
      mpiexec -n 1 ../bin/picongpu -d 1 1 1 -g 660 660 660 --periodic 0 0 1 -s 600 \
        --fields_energy.period 10 \
        --openPMD.period 100 --openPMD.ext bp --openPMD.file simData_ref

    else
      echo "There should be no values for i other than 0,1"
    fi

    cd ..
  fi
done

dataPath=$(absolute_path "$simPath/openPMD")


#################################
## validate simulation results ##
#################################
if [ -d $dataPath ] ; then
    echo "Validate!"
    ./bin/validate.sh -d "$dataPath"
fi
ret=$?

exit $ret
