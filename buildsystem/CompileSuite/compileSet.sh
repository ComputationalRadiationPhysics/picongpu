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
# compile a specific set of an example
#
# $1: example name ($example_name)
# $2: cmakePreset number
# $3: globalCMakeOptions
# $4: tmp dir we use ($tmpRun_path)
# $5: build dir in the tmp folder
# $6: examples dir ($examples_path)
# $7: quiet run ($quiet_run)
#

cS_this_dir=$(cd `dirname $0` && pwd)

# load libs and functions ######################################################
#
. $cS_this_dir/color.sh
. $cS_this_dir/exec_helper.sh

# parse options ################################################################
#
    cS_example_name="$1"
    cS_testFlagNr="$2"
    cS_globalCMakeOptions="$3"

    cS_tmpRun_path="$4"
    cS_buildDir="$5"
    cS_examples_path="$6"

    quiet_run="$7"

    # Do not add the example name again, if we compile a single example only
    if [ "$cS_example_name" == "`basename $cS_examples_path`" ] ; then
        cS_examples_path="$cS_examples_path/.."
    fi

# return code of this script (globals) #########################################
#
    myError=0
    myErrorTxt=""

# exec #########################################################################
#
    cd $cS_buildDir

    param_folder="$cS_tmpRun_path/params/$cS_example_name/cmakePreset_$cS_testFlagNr"
    execute_and_validate $cS_this_dir/../../bin/pic-create -f $cS_examples_path/$cS_example_name $param_folder

    execute_and_validate $cS_this_dir/../../bin/pic-configure $cS_globalCMakeOptions -t $cS_testFlagNr $param_folder
    execute_and_validate make install

    echo "$myError" > ./returnCode
    echo "$myErrorTxt" > ./returnTxt

    # go back from "cd $cS_buildDir"
    cd -
