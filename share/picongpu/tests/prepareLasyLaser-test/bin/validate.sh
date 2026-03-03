#!/bin/bash
#
# This file is part of PIConGPU.
# Copyright 2026-2026 PIConGPU contributors
# Authors: Edgar Marquardt
# License: GPLv3+
#
help()
{
  echo "This does nothing"
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

echo "Nothing to validate"

exit 0
