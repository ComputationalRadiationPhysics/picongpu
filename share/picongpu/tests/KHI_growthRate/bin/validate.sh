#!/bin/bash
#
# This file is part of PIConGPU.
# Copyright 2022 PIConGPU contributors
# Authors: Mika Soren Voss
# License: GPLv3+
#

while [[ $# > 0 ]] ; do
        case "$1" in

                -t|--testPar)
                        testPar="$2"
                        shift
                        ;;

                -s|--simDir)
                        simDir="$2"
                        shift
                        ;;

                --help|-h)
                        echo "With this file the test of the growth rate of the KHI"
                        echo "can be started. An already existing KHI simulation"
                        echo "must be used."
                        echo "Alternatives to run the test suite are ci.sh and MainTest.py."
                        echo "If the parameter -t is not set, it is assumed that"
                        echo "picongpu is in the home directory"
                        echo "Usage:"
                        echo "  --testPar|-t \"direction/to/picongpu/test/KHI_growthRate/\""
                        echo "  --simDir|-s \"direction/to/simulation\""
                        echo "  --help|-h"
                        exit 0
                        ;;
        esac
        shift
done

if [ -z "$simDir" ] ; then
        simDir="."
fi

if [ -z "$testPar" ] ; then
        testPar=$HOME
fi

# test for growth rate
MAINTEST="$testPar/picongpu/lib/python/test/KHI_growthRate"

python $MAINTEST/MainTest.py $testPar/picongpu/test/KHI_growthRate/include/picongpu/param/ $simDir/simOutput/
exit $?
