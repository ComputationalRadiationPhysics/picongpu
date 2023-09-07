#!/bin/bash
#
# This file is part of PIConGPU.
# Copyright 2022-2023 PIConGPU contributors
# Authors: Mika Soren Voss, Rene Widera
# License: GPLv3+
#

function absolute_path()
{
    cd $1
    pwd
}

help()
{
  echo "Execute the KHI example and validate the output data."
  echo "The validation is evaluating the magnetic field growth rate with the corresponding analytic solution."
  echo ""
  echo "Usage:"
  echo "    ci.sh [-d dataPath] [inputSetPath] [destinationPath]"
  echo ""
  echo "  -d | --data dataPath                - path to simulation output data"
  echo "  --delete                            - delete dataPath after verification"
  echo "                                        Default: destinationPath/simOutput_<timestamp>"
  echo "  -f | --force                        - Override if 'destinationPath' exists."
  echo "  -h | --help                         - show help"
  echo ""
  echo "  inputSetPath                         - path to the simulation input set"
  echo "                                         Default: current directory"
  echo "  destinationPath                      - path to the destination where the input set is cloned to via"
  echo "                                         'pic-create'"
  echo "                                         Default: current directory"
}

currentPath=$(cd `dirname $0` && pwd)
currentPath=$(absolute_path $currentPath)


# options may be followed by
# - one colon to indicate they has a required argument
OPTS=`getopt -o d:hf -l data:,help,force,delete -- "$@"`
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
        -f|--force)
            force="true"
            ;;
        --delete)
            deleteData="true"
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
if [ $# -ge 1 ] ; then
    inputSetPath="$1"
else
    inputSetPath="./"
fi

echo $*
if [ $# -eq 2 ] ; then
    inputDestinationPath=$2
    create="true"
else
    inputDestinationPath=$inputSetPath
fi

if [ -z "$dataPath" ] ; then
    date_stamp=$(date +"%F-%h-%M-%S")
    dataPath="$inputDestinationPath/simOutput_$date_stamp"
fi

if [ -d "$dataPath" ] ; then
    if [ "$force" == "true" ] ; then
        echo "Warning: using existing folder on user-request [-f]"
    else
        echo "Destination path already in use, cannot create new folder" >&2
        exit 1
    fi
fi

if [ "$create" == "true" ] ; then
    if [ "$force" == "true" ] ; then
        pic-create -f $inputSetPath $inputDestinationPath
    else
        pic-create $inputSetPath $inputDestinationPath
    fi
fi

mkdir -p $dataPath

# use absolut path's
inputDestinationPath=$(absolute_path $inputDestinationPath)
dataPath=$(absolute_path $dataPath)

cd $inputDestinationPath
pic-build

cd $dataPath
mpiexec -n 1 $inputDestinationPath/bin/picongpu -d 1 1 1 -g 192 512 12 --periodic 1 1 1 -s 3000 --fields_energy.period 10
cd ..

# test with python
$currentPath/validate.sh $inputDestinationPath -d $dataPath
ret=$?

if [ "$deleteData" == "true" ] ; then
        rm -r $dataPath
fi

exit $ret
