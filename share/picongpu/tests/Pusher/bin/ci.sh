#!/bin/bash
#
# This file is part of PIConGPU.
# Copyright 2023 PIConGPU contributors
# Authors: Mika Soren Voss, Hannes Wolf, Klaus Steiniger
# License: GPLv3+
#

function absolute_path()
{
    builtin cd -- $1 && pwd
}

help()
{
  echo "Execute a single particle simulation in a homogenious magnetic field to validate the pusher."
  echo "Simulation results are validated against by checking the change of radius during the simulation and the quantitative phase shift"
  echo ""
  echo "Usage: ci.sh"
  echo ""
  echo "Options"
  echo "-t | --flag                   - flag number of cmakeFlags"
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
OPTS=`getopt -ot:h help flag: -- "$@"`
if [ $? != 0 ] ; then
    # something went wrong, getopt will put out an error message for us
    exit 1
fi

eval set -- "$OPTS"

# standard value of -t|--flag:
declare -i flag=0

# parser
while true ; do
    case "$1" in
        -t|--flag)
            declare -i flag=$2
            shift
            ;;
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
if [ -d "./include" ] ; then
  pic-build -t $flag
  ret_build=$?

else
  echo "Execute ci.sh from the directory where the simulation include dir is located!"
  exit 1
fi

if [ $ret_build -eq 0 ] ; then
  ## create simulation data directory
  date_stamp=$(date +"%F-%h-%M-%S")
  simPath="./simOutput_$date_stamp"

  if [ -d "$simPath" ] ; then
      echo "Destination path already in use, cannot create new folder" >&2
      exit 1
  fi

  mkdir -p $simPath

  # use absolut path's
  simPath=$(absolute_path $simPath)

  cd $simPath
  echo "Run!"
  mpiexec -n 1 ../bin/picongpu -d 1 1 1 -g 64 64 32 -s 100 --fieldBackground.duplicateFields --periodic 1 1 1 --openPMD.period 1 --openPMD.file simData --openPMD.ext bp
  cd ..

  dataPath=$(absolute_path "$simPath/openPMD")
else
  echo "could not build sumulation"
  exit 2

fi


#################################
## validate simulation results ##
#################################
if [ -d $dataPath ] ; then
    echo "Validate!"
    ./bin/validate.sh -d "$dataPath/simData_%T.bp"
else
      echo "dataPath does not exist"
      exit 3
fi

ret=$?

exit $ret


