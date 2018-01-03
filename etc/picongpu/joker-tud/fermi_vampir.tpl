#!/usr/bin/env bash
# Copyright 2013-2018 Axel Huebl, Rene Widera, Richard Pausch
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
.TBG_profile=${PIC_PROFILE:-"~/picongpu.profile"}

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
module load vampirtrace/gpu-gnu-cuda5.0

unset MODULES_NO_OUTPUT

export VT_MPI_IGNORE_FILTER=yes
export VT_PFORM_GDIR=traces
export VT_FILE_PREFIX=trace
export VT_BUFFER_SIZE=3G
export VT_MAX_FLUSHES=3
export VT_GNU_DEMANGLE=yes
export VT_PTHREAD_REUSE=yes
export VT_FILTER_SPEC=!TBG_dstPath/tbg/cuda.filter
export VT_UNIFY=yes
export VT_GPUTRACE=yes,cupti,idle,memusage,concurrent
export VT_VERBOSE=1
export VT_CUPTI_METRICS=
export VT_CUDATRACE_BUFFER_SIZE=200M


mkdir simOutput 2> /dev/null
cd simOutput

# test if cuda_memtest binary is available
if [ -f !TBG_dstPath/input/bin/cuda_memtest ] ; then
  $MPI_ROOT/bin/mpirun !TBG_dstPath/input/bin/cuda_memtest.sh
else
  echo "no binary 'cuda_memtest' available, skip GPU memory test" >&2
fi

if [ $? -eq 0 ] ; then
   $MPI_ROOT/bin/mpirun -x VT_MPI_IGNORE_FILTER -x VT_PFORM_GDIR -x VT_FILE_PREFIX -x VT_BUFFER_SIZE -x VT_MAX_FLUSHES -x VT_GNU_DEMANGLE -x VT_PTHREAD_REUSE -x VT_FILTER_SPEC -x VT_UNIFY -x VT_GPUTRACE -x VT_VERBOSE -x VT_CUPTI_METRICS -x VT_CUDATRACE_BUFFER_SIZE --display-map -am !TBG_dstPath/tbg/openib.conf --mca mpi_leave_pinned 0 !TBG_dstPath/input/bin/picongpu !TBG_author !TBG_programParams | tee output
fi

