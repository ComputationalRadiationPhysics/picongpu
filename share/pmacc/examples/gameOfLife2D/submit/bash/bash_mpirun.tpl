#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Rene Widera, Axel Huebl
#
# SPDX-License-Identifier: LGPL-3.0-or-later
#

##calculations will be performed by tbg##

# settings that can be controlled by environment variables before submit
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}

# 4 gpus per node if we need more than 4 gpus else same count as TBG_tasks
.TBG_gpusPerNode=$(if [ $TBG_tasks -gt 4 ] ; then echo 4; else echo $TBG_tasks; fi)

## end calculations ##


echo 'Running program...'

cd !TBG_dstPath

export MODULES_NO_OUTPUT=1
. ~/picongpu.profile
unset MODULES_NO_OUTPUT

#set user rights to u=rwx;g=r-x;o=---
umask 0027

mkdir simOutput 2> /dev/null
cd simOutput

mpirun  -tag-output --display-map -npernode !TBG_gpusPerNode -n !TBG_tasks !TBG_dstPath/bin/!TBG_PROGRAM !TBG_author | tee output
