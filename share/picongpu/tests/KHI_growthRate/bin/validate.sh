#!/bin/bash
#
# This file is part of PIConGPU.
# Copyright 2022-2023 PIConGPU contributors
# Authors: Mika Soren Voss, Rene Widera
# License: GPLv3+
#

help()
{
  echo "Validate KHI output data."
  echo "The test is evaluating the magnetic field growth rate with the corresponding analytic solution."
  echo ""
  echo "Usage:"
  echo "    validate.sh [-d dataPath] [inputSetPath]"
  echo ""
  echo "  -d | --data dataPath                 - path to simulation output data"
  echo "                                         Default: inputPath/simOutput"
  echo "  -h | --help                          - show help"
  echo ""
  echo "  inputSetPath                         - path to the simulation input set"
  echo "                                         Default: current directory"
}

# options may be followed by
# - one colon to indicate they has a required argument
OPTS=`getopt -o d:h -l data:,help -- "$@"`
if [ $? != 0 ] ; then
    # something went wrong, getopt will put out an error message for us
    exit 1
fi

eval set -- "$OPTS"

# parser
while true ; do
    case "$1" in
        -d|--data)
            dataPath=$2
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


# the first parameter is the project path
if [ $# -eq 1 ] ; then
    inputSetPath="$1"
else
    inputSetPath="./"
fi

if [ -z "$dataPath" ] ; then
    dataPath=$inputSetPath/simOutput
fi

# test for growth rate
MAINTEST="$PICSRC/lib/python/test/setups/ESKHI"

python $MAINTEST/main.py -p "$inputSetPath/include/picongpu/param" -r "$dataPath" -s "$dataPath"
exit $?


