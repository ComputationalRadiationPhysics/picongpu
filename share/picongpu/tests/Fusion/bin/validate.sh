#!/bin/bash
#
# Validation wrapper script for PIConGPU fusion CI tests
#
# This file is part of PIConGPU.
# Copyright 2024 PIConGPU contributors
# Authors: GitHub Copilot, Filip Optolowicz
# License: GPLv3+
#

set -o pipefail

function absolute_path()
{
    builtin cd -- "$1" &> /dev/null && pwd
}

help()
{
  echo "Validation script for fusion CI tests"
  echo ""
  echo "Usage:"
  echo "    validate.sh -d <data_path>"
  echo ""
  echo "Options:"
  echo "-h | --help                   - show help"
  echo "-d | --data <path>           - path to simulation output data"
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
OPTS=`getopt -o hd: -l help,data: -- "$@"`
if [ $? != 0 ] ; then
    # something went wrong, getopt will put out an error message for us
    exit 1
fi

eval set -- "$OPTS"

# initialize variables
dataPath=""

# parser
while true ; do
    case "$1" in
        -h|--help)
            echo -e "$(help)"
            shift
            exit 0
            ;;
        -d|--data)
            dataPath="$2"
            shift 2
            ;;
        --) shift; break;;
    esac
done

############################
## validate simulation ##
############################

if [ -z "$dataPath" ] ; then
    echo "Error: No data path specified. Use -d <path> option."
    exit 1
fi

if [ ! -d "$dataPath" ] ; then
    echo "Error: Data path '$dataPath' does not exist or is not a directory."
    exit 1
fi

echo "Validating simulation results in: $dataPath"

# Run the Python validation script from lib directory
python3 "$currentPath/../lib/validate.py" "$dataPath"
exit $?
