#!/bin/bash
#
# This file is part of PIConGPU.
# Copyright 2023-2026 PIConGPU contributors
# Authors: Mika Soren Voss, Hannes Wolf, Klaus Steiniger, Max Lehmann, Edgar Marquardt
# License: GPLv3+
#

function absolute_path()
{
    builtin cd -- $1 && pwd
}

help()
{
  echo "prepare al laser via Lasy and prepareLasyLaser and import to a simple PIConGPU simulation."
  echo "Tests, whether all this works."
  echo ""
  echo "Usage: ci.sh [-d dataPath] [inputSetPath] [destinationPath]"
  echo ""
  echo "Options"
  echo "-h | --help                 - show help"
  echo ""
  echo "inputSetPath                - path to the simulation input set"
  echo "                              Default: current directory"
  echo "destinationPath             - path to the destination where the input set is cloned to via"
  echo "                              'pic-create'"
  echo "                              Default: current directory"
}

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
if [ $# -eq 2 ] ; then
  inputSetPath=$1
  inputDestinationPath=$2
else
  echo "Two arguments are required, $# given!"
  echo -e "$(help)"
fi

if [ -d "$inputSetPath/include" ] ; then
  if [ -d "$inputDestinationPath" ] ; then
    echo "Output directory $inputDestinationPath exists" >&2
    echo "Please remove" >&2
    exit 1
  fi
  echo "start setting up"
  pic-create $inputSetPath $inputDestinationPath
  inputDestinationPath=$(absolute_path $inputDestinationPath)
  cd $inputDestinationPath
  if [ -d "./lib/python/test/" ] ; then
    echo "create laser file"
    python3 ./lib/python/test/make_laser.py -r $inputDestinationPath
    ret_make=$?
    if [ $ret_make -ne 0] ; then
      echo "creating laser failed" >&2
      exit $ret_make
    fi
  else
    echo "Input path $inputSetPath does not contain an lib/python/test directory" >&2
    exit 2
  fi
  echo "building"
  pic-build
  ret_build=$?

else
  echo "Input path $inputSetPath does not contain an include directory" >&2
  exit 2
fi

if [ $ret_build -eq 0 ] ; then
  ## create simulation data directory
  date_stamp=$(date +"%F-%H-%M-%S")
  simPath="./simOutput_$date_stamp"

  if [ -d "$simPath" ] ; then
      echo "Simulation path already in use, cannot create new folder" >&2
      exit 3
  fi

  mkdir -p $simPath

  # use absolut path's
  simPath=$(absolute_path $simPath)

  cd $simPath

  # run the simulation
  echo "Simulation path: " $simPath"/"
  mpiexec -n 1 ../bin/picongpu -d 1 1 1 -g 64 64 64 --periodic 1 1 1 -s 32 \
  --openPMD.period 8 --openPMD.ext bp --openPMD.file simData

  ret_sim=$?
  if [ $ret_sim -ne 0] ; then
    echo "running simulation failed" >&2
    exit $ret_sim
  fi
else
  echo "build failed" >&2
  exit $ret_build
fi

cd ..
python3 ./lib/python/test/validate_laser.py -r $simPath
ret=$?
if [ $ret -eq 0 ] ; then
  echo "test successfully validated"
else
  echo "test validation failed"
fi
exit $ret
