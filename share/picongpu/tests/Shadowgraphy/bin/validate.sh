#!/bin/bash
#
# This file is part of PIConGPU.
# Copyright 2023 PIConGPU contributors
# Authors: Mika Soren Voss, Rene Widera, Hannes Wolf, Klaus Steiniger, Finn-Ole Carstens
# License: GPLv3+
#

set -o pipefail

# define testSuite variable, set correctly further below if
# buildsystem/CompileSuite/color.sh from PIConGPU is available
testSuite="" # define variable, set correctly

currentDir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
picongpuPrefix=""

if [ ! -z $PICSRC ]; then
    picongpuPrefix="$PICSRC"
elif [ ! -z $CI_PROJECT_DIR ]; then
    picongpuPrefix="$CI_PROJECT_DIR"
fi

printf "PIConGPU prefix = %s\n" $picongpuPrefix

## Load libs and functions to use the thumbs_up and thumbs_down
if [ ! -z $picongpuPrefix ]; then
    if [ -d $picongpuPrefix ]; then
        printf "Load buildsystem/CompileSuite/color.sh from PIConGPU!\n"
        . $picongpuPrefix/buildsystem/CompileSuite/color.sh
        testSuite=`echo_b "[testSuite] "`
    fi
fi

help()
{
  echo "Validate shadowgraphy plugin output data."
  echo ""
  echo "Usage:"
  echo "    validate.sh [-d dataPath] [inputSetPath]"
  echo ""
  echo "  -d | --data dataPath                 - path to simulation output data"
  echo "                                         Default: inputPath/simOutput/shadowgraphy_%T.bp"
  echo "  -h | --help                          - show help"
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

MAINTEST="./lib/python/test/Shadowgraphy"

if [ -z "$dataPath" ]; then
    dataPath="$( dirname -- "${currentDir}" )/simOutput/"
    if [ ! -d "$dataPath" ]; then
        echo "Directory $dataPath not existent."
        echo "Please provide path to data with option -d!"
        exit 1
    fi
fi

printf "Execute validate.py with data from %s\n" $dataPath
python3 $MAINTEST/validate.py $dataPath
test_return=$?

if [ ! -z $picongpuPrefix ]; then
    if [ $test_return -ne 0 ]; then
        echo -e $testSuite`echo_r "Shadowgraphy plugin has bad performance!"`
        echo -e $testSuite`echo_r "Check shadowgraphy plugin implementation!"`
        echo -e $testSuite"`thumbs_down`"
    else
        echo -e $testSuite`echo_g "Shadowgraphy plugin implementation successfully validated!"`
        echo -e $testSuite"`thumbs_up`"
    fi
fi

exit $test_return
