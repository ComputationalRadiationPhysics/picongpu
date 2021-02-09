#!/usr/bin/env bash
#
# Copyright 2013-2021 Axel Huebl
#
# This file is part of PIConGPU.
#
# PIConGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PIConGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.
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
