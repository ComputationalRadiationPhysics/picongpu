#!/bin/bash
#
# This file is part of PIConGPU.
# Copyright 2023 PIConGPU contributors
# Authors: Mika Soren Voss, Rene Widera, Hannes Wolf
# License: GPLv3+
#
help()
{
  echo "Validate SingleParticle output data."
  echo "The test is evaluating the change of Radius of the Pusher and the qualitative phase shift"
  echo ""
  echo "Usage:"
  echo "    validate.sh [dataPaths]"
  echo ""
  echo "    dataPaths                - paths to simulation output data"
  echo "                                         Default: inputPath/simOutput/simData_%T.h5"
  echo "  -h | --help                          - show help"
  echo ""
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
        -h|--help)
            echo -e "$(help)"
            shift
            exit 0
            ;;
        --) shift; break;;
    esac
    shift
done

if [ -z "$dataPath" ] ; then
    dataPath=$0/../simOutput/openPMD/simData_%T.bp
fi

MAINTEST="./lib/python/test/PusherScaling"
if [ -d $MAINTEST ] ; then
   python3 $MAINTEST/MainTest.py $@
   exit $?
fi