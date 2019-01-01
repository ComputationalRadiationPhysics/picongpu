#!/usr/bin/env bash
# Copyright 2013-2019 Axel Huebl, Rene Widera, Felix Schmitt
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


# PIConGPU batch script for titan aka jaguar PBS batch system

#PBS -q !TBG_queue
#PBS -l walltime=!TBG_wallTime
# Sets batch job's name
#PBS -N !TBG_jobName
#PBS -l nodes=!TBG_nodes
# send me mails on job (b)egin, (e)nd, (a)bortion or (n)o mail
#PBS -m !TBG_mailSettings -M !TBG_mailAddress
#PBS -d !TBG_dstPath
#PBS -A !TBG_nameProject

#PBS -o stdout
#PBS -e stderr

#PBS -l gres=atlas1%atlas2


## calculations will be performed by tbg ##
.TBG_queue="batch"

# settings that can be controlled by environment variables before submit
.TBG_mailSettings=${MY_MAILNOTIFY:-"n"}
.TBG_mailAddress=${MY_MAIL:-"someone@example.com"}
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}
.TBG_nameProject=${proj:-""}
.TBG_profile=${PIC_PROFILE:-"${PROJWORK}/!TBG_nameProject/${USER}/picongpu.profile"}

# use ceil to caculate nodes
.TBG_nodes=!TBG_tasks
## end calculations ##

echo 'Running program...'
echo !TBG_jobName

cd !TBG_dstPath
echo -n "present working directory:"
pwd

source !TBG_profile 2>/dev/null
if [ $? -ne 0 ] ; then
  echo "Error: PIConGPU environment profile under \"!TBG_profile\" not found!"
  exit 1
fi

export SCOREP_ENABLE_PROFILING=no
export SCOREP_ENABLE_TRACING=yes
export SCOREP_EXPERIMENT_DIRECTORY=traces
export SCOREP_CUDA_ENABLE=yes,flushatexit
export SCOREP_CUDA_BUFFER=200M
export SCOREP_TOTAL_MEMORY=1G
export SCOREP_FILTERING_FILE=!TBG_dstPath/tbg/scorep.filter

mkdir simOutput 2> /dev/null
cd simOutput

#aprun  -N 1 -n !TBG_nodes !TBG_dstPath/input/bin/cuda_memtest.sh

#if [ $? -eq 0 ] ; then
aprun  -N 1 -n !TBG_nodes  !TBG_dstPath/input/bin/picongpu !TBG_author !TBG_programParams | tee output
#fi
