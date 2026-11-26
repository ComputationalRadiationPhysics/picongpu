#!/bin/bash
#
# This file is part of PIConGPU.
# Copyright 2023-2025 PIConGPU contributors
# Authors: Mika Soren Voss, Hannes Wolf, Klaus Steiniger, Filip Optolowicz
# License: GPLv3+
#

set -o pipefail

function absolute_path()
{
    builtin cd -- "$1" &> /dev/null && pwd
}

help()
{
  echo "Simulate a bunch of Deuterons and Tritons moving in opposite directions and fusing."
  echo "Generated number of He4 particles is compared to analytical prediction."
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
else
  mkdir -p $simPath
  echo "Created simulation output directory $simPath"
fi

ret_params=$?
if [ $ret_params -ne 0 ] ; then
  echo "Error changing parameters randomly"
  exit 1
fi


# use absolut path's
simPath=$(absolute_path $simPath)

echo "Build setup!"
pwd
echo
echo
pic-build
ret_build=$?
if [ $ret_build -eq 0 ] ; then
  cd $simPath
  echo "Run setup!"

  # Energy histogram settings [in keV]
  TBG_t_Bin="--t_energyHistogram.period 10 --t_energyHistogram.filter all --t_energyHistogram.binCount 3 --t_energyHistogram.minEnergy 0 --t_energyHistogram.maxEnergy 2"
  TBG_d_Bin="--d_energyHistogram.period 10 --d_energyHistogram.filter all --d_energyHistogram.binCount 3 --d_energyHistogram.minEnergy 0 --d_energyHistogram.maxEnergy 2"
  TBG_n_Bin="--n_energyHistogram.period 10 --n_energyHistogram.filter all --n_energyHistogram.binCount 3 --n_energyHistogram.minEnergy 0 --n_energyHistogram.maxEnergy 2"
  TBG_He4_Bin="--He4_energyHistogram.period 10 --He4_energyHistogram.filter all --He4_energyHistogram.binCount 3 --He4_energyHistogram.minEnergy 0 --He4_energyHistogram.maxEnergy 2"

  # run the sim: 1 node, 24x24x24 cells, 250 steps
  mpiexec -n 1 ../bin/picongpu -d 1 1 1 -g 24 24 24 --periodic 1 1 1 -s 500 \
    --d_macroParticlesCount.period 10 --t_macroParticlesCount.period 10 \
    --n_macroParticlesCount.period 10 --He4_macroParticlesCount.period 10 \
    $TBG_t_Bin $TBG_d_Bin $TBG_n_Bin $TBG_He4_Bin
  cd ..
fi

dataPath=$(absolute_path "$simPath/")


#################################
## validate simulation results ##
#################################
if [ -d $dataPath ] ; then
    echo "Validate!"
    ./bin/validate.sh -d "$dataPath"
fi
ret=$?
exit $ret
