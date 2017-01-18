#!/usr/bin/env bash
# Copyright 2013-2017 Axel Huebl, Rene Widera, Richard Pausch
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


# PIConGPU batch script for joker PBS PRO batch system

#PBS -q !TBG_queue
#PBS -l walltime=!TBG_wallTime

# Sets batch job's name
#PBS -N !TBG_jobNameShort
#PBS -l select=!TBG_nodes:mpiprocs=!TBG_gpusPerNode:ncpus=!TBG_coresPerNode:ngpus=!TBG_gpusPerNode:gputype=!TBG_gpu_arch -lplace=excl

# send me mails on job (b)egin, (e)nd, (a)bortion or (n)o mail
#PBS -m TBG_mailSettings
#PBS -M TBG_mailAddress

#PBS -o !TBG_dstPath/stdout
#PBS -e !TBG_dstPath/stderr


## calculation are done by tbg ##
.TBG_gpu_arch="fermi"
.TBG_queue="workq"

# settings that can be controlled by environment variables before submit
.TBG_mailSettings=${MY_MAILNOTIFY:-"n"}
.TBG_mailAddress=${MY_MAIL:-"someone@example.com"}
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}

# 4 gpus per node if we need more than 4 gpus else same count as TBG_tasks
.TBG_gpusPerNode=`if [ $TBG_tasks -gt 4 ] ; then echo 4; else echo $TBG_tasks; fi`

# use one core per gpu
.TBG_coresPerNode=$TBG_gpusPerNode

# use ceil to caculate nodes
.TBG_nodes="$(( ( TBG_tasks + TBG_gpusPerNode -1 ) / TBG_gpusPerNode))"
## end calculations ##

echo 'Running program...'
echo !TBG_jobName

cd !TBG_dstPath
echo -n "present working directory:"
pwd


export MODULES_NO_OUTPUT=1

. /etc/profile.d/modules.sh
module add shared software boost cupti papi/4.2.0  cuda/5.0.35 gdb pngwriter cmake gdb hdf5/1.8.5-threadsafe 2>/dev/null
module load gcc/4.6.2 openmpi/1.6.2-gnu

unset MODULES_NO_OUTPUT


mkdir simOutput 2> /dev/null
cd simOutput

$MPI_ROOT/bin/mpirun !TBG_dstPath/picongpu/bin/cuda_memtest.sh

if [ $? -eq 0 ] ; then
   $MPI_ROOT/bin/mpirun  --display-map -am !TBG_dstPath/tbg/openib.conf --mca mpi_leave_pinned 0 !TBG_dstPath/picongpu/bin/picongpu !TBG_author !TBG_programParams | tee output
fi
