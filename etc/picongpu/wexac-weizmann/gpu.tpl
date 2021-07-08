#!/usr/bin/env bash
# Copyright 2013-2021 Axel Huebl, Anton Helm, Richard Pausch, Rene Widera,
#                     Marco Garten
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


# PIConGPU batch script for wexacs's BSUB batch system
#BSUB -csm y
#BSUB -q !TBG_queue
#BSUB -m !TBG_gpuType
#BSUB -W !TBG_wallTimeNoSeconds

# sets batch job's name
#BSUB -J !TBG_jobName
#BSUB -n !TBG_tasks
#BSUB -R rusage[mem=48000]
#BSUB !TBG_mailSettings -u !TBG_mailAddress
#BSUB -cwd !TBG_dstPath

# sets output
#BSUB -o stdout.%J
#BSUB -e stderr.%J


## calculations will be performed by tbg
# extract queue and gpu selection
.TBG_queue=${TBG_partition:-"gpu-short"}
.TBG_gpuType=${TBG_gpuType:-"dgx_hosts"}

# remove seconds from walltime
.TBG_wallTimeNoSeconds=${TBG_wallTime::-3}

# settings that can be controlled by environment variables before submit
.TBG_mailSettings=${MY_MAILNOTIFY:-""}
.TBG_mailAddress=${MY_MAIL:-"someone@example.com"}
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}
.TBG_profile=${PIC_PROFILE:-"~/picongpu.profile"}

# number of available/hosted GPUs per node in the system
.TBG_numHostedGPUPerNode=8

# required GPUs per node for the current job
.TBG_gpusPerNode=`if [ $TBG_tasks -gt $TBG_numHostedGPUPerNode ] ; then echo $TBG_numHostedGPUPerNode; else echo $TBG_tasks; fi`

# number of cores to block per GPU - we got 7 cpus per gpu
#   and we will be accounted 7 CPUs per GPU anyway
.TBG_coresPerGPU=7

# use ceil to caculate nodes
.TBG_nodes="$((( TBG_tasks + TBG_gpusPerNode - 1 ) / TBG_gpusPerNode))"

## end calculations ##

echo 'Running program...'

cd !TBG_dstPath

export MODULES_NO_OUTPUT=1
source !TBG_profile
if [ $? -ne 0 ] ; then
  echo "Error: PIConGPU environment profile under \"!TBG_profile\" not found!"
  exit 1
fi
unset MODULES_NO_OUTPUT

#set user rights to u=rwx;g=r-x;o=---
umask 0027

mkdir simOutput 2> /dev/null
cd simOutput

export OMP_NUM_THREADS=!TBG_coresPerGPU
mpiexec -n !TBG_tasks !TBG_dstPath/input/bin/picongpu !TBG_author !TBG_programParams | tee output
# note: instead of the PIConGPU binary, one can also debug starting "js_task_info | sort"
