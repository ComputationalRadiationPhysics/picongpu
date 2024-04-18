#!/bin/bash
#
# This file is part of PIConGPU.
# Copyright 2023 PIConGPU contributors
# Authors: Mika Soren Voss, Hannes Wolf, Klaus Steiniger, Finn-Ole Carstens
# License: GPLv3+
#

set -o pipefail

function absolute_path()
{
    builtin cd -- "$1" &> /dev/null && pwd
}

help()
{
  echo "Simulate a Gaussian pulse propagating in z direction to quantify the Shadowgraphy plugin performance"
  echo "the following values to the expectation value: energy in shadowgram, position of peak in shadowgram,"
  echo "width of Gaussian pulse, bandwidth of all field components, and mean frequency of all field components."
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

pic-build
ret_build=$?

if [ $ret_build -eq 0 ] ; then
  cd $simPath
  echo "Run shadowgraphy simulation"
  mpiexec -n 1 ../bin/picongpu -d 1 1 1 -g 208 208 64 -s 1865 \
    --shadowgraphy.start 625 --shadowgraphy.file shadowgraphy --shadowgraphy.slicePoint 0.5 \
    --shadowgraphy.focusPos 0 --shadowgraphy.duration 1230 --shadowgraphy.fourierOutput true
  cd ..
fi

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
