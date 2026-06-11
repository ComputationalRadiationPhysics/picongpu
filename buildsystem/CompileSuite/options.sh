#!/usr/bin/env bash

# SPDX-FileCopyrightText: Axel Huebl
#
# SPDX-License-Identifier: GPL-3.0-or-later

#
#

function parseOptions()
{
# options may be followed by one colon to indicate they have a required argument
OPTS=`getopt -o lqj:c:h -l help,cmake: -- "$@"`
if [ $? != 0 ] ; then
    # something went wrong, getopt will put out an error message for us
    exit 1
fi

eval set -- "$OPTS"

while true ; do
    case "$1" in
        -l)
            list_param=1
            ;;
        -q)
            quiet_run=1
            ;;
        -j)
            num_parallel="$2"
            quiet_run=1
            shift
            ;;
        -c|--cmake)
            globalCMakeOptions="$2"
            shift
            ;;
        -h|--help)
            echo -e "$(help)"
            exit 1
            ;;
        --) shift; break;;
    esac
    shift
done

examples_path="$1"
tmpRun_path="$2"

}
