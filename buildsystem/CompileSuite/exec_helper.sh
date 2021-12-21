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

# $1 is the error code
# $2 is the text to be checked
function validate_error()
{
    if [ $1 -ne 0 ] ; then
        echo -e $compileSuite`echo_r "ERROR: $2"` >&2

        # add information about the specific CMake env
        myExt=""
        myInstall=""
        if [ -f "./CMakeCache.txt" ] ; then
            echo -e $compileSuite"`echo_r 'cmake -L .' && cmake -L .`" >&2
            myExt=`cmake -L . | grep "PIC_EXTENSION_PATH"`
            myInstall=`cmake -L . | grep "CMAKE_INSTALL_PREFIX"`
        fi

        myError=$1
        myErrorTxt="$myErrorTxt\n$compileSuite$compileError"
        myErrorTxt="$myErrorTxt""In "`echo_r $myExt`": "$myInstall" ("`pwd`") $2"

        if [ $quiet_run -ne 1 ] ; then
            read -e -p "Ignore and run next [yes|NO]? : " input
            if [ -z "$input" ] ; then
                input="NO"
            fi
            input=`echo "$input" | tr '[:upper:]' '[:lower:]'`
            if [ $input != "yes" ] ; then
                echo "Exit..." >&2
                exit 1
            fi
        fi
    fi
}

# $1 command
function execute_and_validate()
{
    echo -e $compileSuite"execute: $*" 1>&2
    eval "$*"
    validate_error "$?" "$*"
}
