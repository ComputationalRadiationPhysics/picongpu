#!/usr/bin/env zsh
# SPDX-FileCopyrightText: Axel Huebl, Anton Helm, Rene Widera, Julian Lenz
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

##calculations will be performed by tbg##

# settings that can be controlled by environment variables before submit
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}
.TBG_profile=${PIC_PROFILE:-"~/picongpu.profile"}

# 8 gpus per node if we need more than 8 gpus else same count as TBG_tasks
.TBG_gpusPerNode=$(if [ $TBG_tasks -gt 8 ] ; then echo 8; else echo $TBG_tasks; fi)

## end calculations ##


echo 'Running program...'

TBG_dstPath="!TBG_dstPath"
cd $TBG_dstPath

export MODULES_NO_OUTPUT=1
. !TBG_profile
unset MODULES_NO_OUTPUT

#set user rights to u=rwx;g=r-x;o=---
umask 0027

mkdir simOutput 2> /dev/null
cd simOutput

# test if cuda_memtest binary is available
if [ -f $TBG_dstPath/input/bin/cuda_memtest ] ; then
  mpirun --mca mpi_leave_pinned 0 -npernode !TBG_gpusPerNode -n !TBG_tasks $TBG_dstPath/input/bin/cuda_memtest.sh
else
  echo "Note: GPU memory test was skipped as no binary 'cuda_memtest' available. This does not affect PIConGPU, starting it now" >&2
fi

if [ $? -eq 0 ] ; then
  mpirun --mca mpi_leave_pinned 0 -npernode !TBG_gpusPerNode -n !TBG_tasks $TBG_dstPath/input/bin/picongpu !TBG_author !TBG_programParams | tee output
fi
