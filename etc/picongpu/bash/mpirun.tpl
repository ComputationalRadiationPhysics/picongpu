#!/usr/bin/env bash
# Copyright 2013-2024 Axel Huebl, Anton Helm, Rene Widera
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.
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

EXE="$TBG_dstPath/input/bin/picongpu"
retry_count=0
while [ ! -f "$EXE" ] && [ $retry_count -lt 10 ]; do
  retry_count=$((retry_count + 1))
  echo "Waiting for $EXE to be available (attempt $retry_count of 10)..."
  sleep 30
done

if [ ! -f "$EXE" ]; then
  echo "Error: $EXE was not found after $retry_count attempts" >&2
  exit 1
fi

# test if cuda_memtest binary is available
if [ -f $TBG_dstPath/input/bin/cuda_memtest ] ; then
  mpirun --mca mpi_leave_pinned 0 -npernode !TBG_gpusPerNode -n !TBG_tasks $TBG_dstPath/input/bin/cuda_memtest.sh
else
  echo "Note: GPU memory test was skipped as no binary 'cuda_memtest' available. This does not affect PIConGPU, starting it now" >&2
fi

if [ $? -eq 0 ] ; then
  mpirun --mca mpi_leave_pinned 0 -npernode !TBG_gpusPerNode -n !TBG_tasks "$EXE" !TBG_author !TBG_programParams | tee output
fi
