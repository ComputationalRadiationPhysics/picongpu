#!/bin/bash
#
# This file is part of PIConGPU.
# Copyright 2023 PIConGPU contributors
# Authors: Mika Soren Voss, Rene Widera, Hannes Wolf
# License: GPLv3+
#
help()
{
  echo "Validate CurrentDeposition output data."
  echo "The test is evaluating the current density field with the corresponding analytic solution."
  echo ""
  echo "Usage:"
  echo "    validate.sh [dataPath]"
  echo ""
  echo "  -d | --data dataPath                 - path to simulation output data"
  echo "                                         Default: inputPath/simOutput/simData_%T.h5"
  echo "  -h | --help                          - show help"
  echo ""
}

# options may be followed by
# - one colon to indicate they have a required argument
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
echo "The data is at $dataPath"
if [ -z "$dataPath" ] ; then
    dataPath=$0/../simOutput/simData_%T.h5
    echo "dataPath did not exist until now \n"
    echo "dataPath has been assigned to $dataPath"
fi

MAINTEST="./lib/python/test/CurrentDeposition"

if [ -d $MAINTEST ] ; then
     echo ""
     python3 $MAINTEST/MainTest.py $dataPath
     exit $?
fi
