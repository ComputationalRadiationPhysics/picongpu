#!/usr/bin/env bash

# SPDX-FileCopyrightText: Axel Huebl
#
# SPDX-License-Identifier: GPL-3.0-or-later

#
#
# compile a specific set of an example
#
# $1: example name ($example_name)
# $2: cmakePreset number: -1 means we do not have presets
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

    caseId=$cS_testFlagNr;
    if [ $caseId -eq -1 ] ; then
        # cmakeFlags file is not available, do not use '-t' option
        caseId=0;
    else
        # cmakeFlags file is available '-t` option can be used
        caseOption="-t $caseId"
    fi

    param_folder="$cS_tmpRun_path/params/$cS_example_name/cmakePreset_$caseId"
    execute_and_validate $cS_this_dir/../../bin/pic-create -f $cS_examples_path/$cS_example_name $param_folder

    execute_and_validate $cS_this_dir/../../bin/pic-configure $cS_globalCMakeOptions $caseOption $param_folder
    execute_and_validate make install

    echo "$myError" > ./returnCode
    echo "$myErrorTxt" > ./returnTxt

    # go back from "cd $cS_buildDir"
    cd -
