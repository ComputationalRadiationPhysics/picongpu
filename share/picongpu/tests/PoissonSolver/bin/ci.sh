#!/bin/bash
#
# This file is part of PIConGPU.
#
# Copyright 2026-2026 PIConGPU contributors
# Author: Edgar Marquardt
#
# PIConGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PIConGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.
#

function absolute_path()
{
    builtin cd -- $1 && pwd
}

help()
{
  echo "Tests the Poisson solver with a simple setup."
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
  echo "Two arguments are required, $# given!" >&2
  echo -e "$(help)" >&2
  exit 1
fi

if [ -d "$inputSetPath/include" ] ; then
  if [ -d "$inputDestinationPath" ] ; then
    echo "Output directory $inputDestinationPath exists" >&2
    echo "removing now" >&2
    rm -r $inputDestinationPath
  fi
  echo "start setting up"
  pic-create $inputSetPath $inputDestinationPath
  ret_create=$?
  if [ $ret_create -ne 0 ] ; then
    echo "pic-create failed"
    exit $ret_create
  fi

  inputDestinationPath=$(absolute_path $inputDestinationPath)
  cd $inputDestinationPath

  echo "building"
  pic-build
  ret_build=$?

else
  echo "Input path $inputSetPath does not contain an include directory" >&2
  exit 2
fi

if [ $ret_build -eq 0 ] ; then
  ## create simulation data directory
  simPath="./simOutput"

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
  mpiexec -n 1 ../bin/picongpu -d 1 1 1 -g 64 64 64 --periodic 1 1 1 -s 4 \
  --openPMD.period 8 --openPMD.ext bp --openPMD.file simData --poisson.activate

  ret_sim=$?
  if [ $ret_sim -ne 0 ] ; then
    echo "running simulation failed" >&2
    exit $ret_sim
  fi

  cd ..
  # validate the results
  if [ -d "./lib/python/test/" ] ; then
    python3 ./lib/python/test/validate_results.py -r $simPath

    ret=$?
    if [ $ret -eq 0 ] ; then
      echo "test successfully validated"
    else
      echo "test validation failed"
    fi
    exit $ret

  else
    echo "Input path $inputSetPath does not contain an lib/python/test directory" >&2
    exit 2
  fi

else
  echo "build failed" >&2
  exit $ret_build
fi
