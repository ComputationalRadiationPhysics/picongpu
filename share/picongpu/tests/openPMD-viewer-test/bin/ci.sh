#!/bin/bash
#
# This file is part of PIConGPU.
# Copyright 2023 PIConGPU contributors
# Authors: Mika Soren Voss, Hannes Wolf, Klaus Steiniger, Max Lehmann
# License: GPLv3+
#

function absolute_path()
{
    builtin cd -- $1 && pwd
}

help()
{
  echo "Execute a simple simulation and add the input data into the dictionary."
  echo "The openPMD-viewer readout is validated against your input data."
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
  cd $inputDestinationPath
  pic-build
  ret_build=$?

else
  echo "Input path $inputSetPath does not contain an include directory" >&2
  exit 2
fi

if [ $ret_build -eq 0 ] ; then
  for backend in bp h5
  do
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
      echo "Run backend " $backend "!"
      echo "Simulation path: " $simPath"/"
      mpiexec -n 1 ../bin/picongpu -d 1 1 1 -g 32 32 12 --periodic 1 1 1 -s 1 \
      --openPMD.period 1 --openPMD.ext $backend --openPMD.file simData

      cd ..

      dataPath=$(absolute_path "$simPath/openPMD")
      #################################
      ## validate simulation results ##
      #################################
      if [ -d $dataPath ] ; then
          echo "Validate backend " $backend " !"
          echo "Data path: " $dataPath"/"
          ./bin/validate.sh -d "$dataPath"
      fi
      ret=$?
      if [ $ret -ne 0 ]; then
          echo "error when validating backend: " $backend
          exit $ret
      fi
  done
fi


exit $ret
