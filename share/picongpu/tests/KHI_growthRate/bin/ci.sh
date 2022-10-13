#!/bin/bash
#
# This file is part of PIConGPU.
# Copyright 2022 PIConGPU contributors
# Authors: Mika Soren Voss
# License: GPLv3+
#

currentPath=$(cd `dirname $0` && pwd)
create="true"

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

                -f|--force)
                        force="true"
                        ;;

                -c|--create)
                        create="false"
                        ;;

                --help|-h)
                        echo "With this file the test of the growth rate of the KHI can"
                        echo " be started. The specified test case is used by default "
                        echo "(see parameters)."
                        echo "Alternatives to run the test suite are validate.sh and MainTest.py."
                        echo "If the parameter -t is not set, it is assumed that"
                        echo " picongpu is in the home directory"
                        echo "Usage:"
                        echo "  --testPar|-t \"direction/to/picongpu/\""
                        echo "  --simDir|-s \"direction/where/simulation/should/be/build/\""
                        echo "  -c|--create do not run pic-create"
                        echo "  -f|--force  - Override if 'destinationPath' exists."
                        echo "  --help|-h"
                        echo "Example:"
                        echo " bash ci.sh - creates a new temporary folder in which the "
                        echo "              test-suite is run and then deletes it"
                        exit 0
                        ;;
        esac
        shift
done

if [ -z "$simDir" ] ; then
        date_stamp=$(date +"%F-%h-%M-%S")
        simDir=/tmp/$date_stamp
fi

if [ -d "$simDir" ] ; then
    if [ "$force" == "true" ] ; then
        echo "Warning: using existing folder on user-request [-f]"
    else
        echo "Destination path already in use, cannot create new folder" >&2
        exit 1
    fi
fi

if [ -z "$testPar" ] ; then
        testPar=$HOME
fi

if [ "$create" == "true" ] ; then
        mktemp -d $date_stamp
        pic-create $testPar/picongpu/test/KHI_growthRate/ $simDir
        currentPath=$simDir/bin/
    else
        mkdir -p $simDir
fi

cd $currentPath/../
pic-build

mkdir $simDir/simOutput

cd $simDir/simOutput
mpiexec -n 1 $currentPath/picongpu -d 1 1 1 -g 192 512 12 --periodic 1 1 1 -s 3000 --fields_energy.period 10
cd ..

# test with python
bash $currentPath/validate.sh -t $testPar -s $simDir
ret=$?

if [ "$create" == "true" ] ; then
        rm -r $simDir
fi

exit $ret
