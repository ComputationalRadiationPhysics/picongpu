#!/usr/bin/env bash
# Copyright 2013-2018 Axel Huebl, Anton Helm, Rene Widera
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


# PIConGPU batch script for hypnos PBS batch system

#PBS -q !TBG_queue
#PBS -l walltime=!TBG_wallTime
# Sets batch job's name
#PBS -N !TBG_jobName
#PBS -l nodes=!TBG_nodes:ppn=!TBG_coresPerNode
#PBS -m !TBG_mailSettings -M !TBG_mailAddress
#PBS -d !TBG_dstPath

#PBS -o stdout
#PBS -e stderr


## calculation are done by tbg ##
.TBG_queue="k20"

# settings that can be controlled by environment variables before submit
.TBG_mailSettings=${MY_MAILNOTIFY:-"n"}
.TBG_mailAddress=${MY_MAIL:-"someone@example.com"}
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}
.TBG_profile=${PIC_PROFILE:-"~/picongpu.profile"}

# number of available/hosted GPUs per node in the system
.TBG_numHostedGPUPerNode=4

# required GPUs per node for the current job
.TBG_gpusPerNode=`if [ $TBG_tasks -gt $TBG_numHostedGPUPerNode ] ; then echo $TBG_numHostedGPUPerNode; else echo $TBG_tasks; fi`

#number of cores per parallel node / default is 2 cores per gpu on k20 queue
.TBG_coresPerNode="$(( TBG_gpusPerNode * 2 ))"

# use ceil to caculate nodes
.TBG_nodes="$(( ( TBG_tasks + TBG_gpusPerNode -1 ) / TBG_gpusPerNode))"
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


#set user rights to u=rwx;g=r-x;o=---
umask 0027

mkdir simOutput 2> /dev/null
cd simOutput

# wait for all nodes to see the output folder
sleep 1

# The OMPIO backend in OpenMPI up to 3.1.3 and 4.0.0 is broken, use the
# fallback ROMIO backend instead.
#   see bug https://github.com/open-mpi/ompi/issues/6285
export OMPI_MCA_io=^ompio

# test if cuda_memtest binary is available and we have the node exclusive
if [ -f !TBG_dstPath/input/bin/cuda_memtest ] && [ !TBG_numHostedGPUPerNode -eq !TBG_gpusPerNode ] ; then
  mpiexec --prefix $MPIHOME -tag-output --display-map -x LIBRARY_PATH -am !TBG_dstPath/tbg/openib.conf --mca mpi_leave_pinned 0 -npernode !TBG_gpusPerNode -n !TBG_tasks !TBG_dstPath/input/bin/cuda_memtest.sh
else
  echo "no binary 'cuda_memtest' available or compute node is not exclusively allocated, skip GPU memory test" >&2
fi

if [ $? -eq 0 ] ; then
  mpiexec --prefix $MPIHOME -x LIBRARY_PATH -x VT_MPI_IGNORE_FILTER -x VT_PFORM_GDIR -x VT_FILE_PREFIX -x VT_BUFFER_SIZE -x VT_MAX_FLUSHES -x VT_GNU_DEMANGLE -x VT_PTHREAD_REUSE -x VT_FILTER_SPEC -x VT_UNIFY -x VT_GPUTRACE -x VT_VERBOSE -x VT_CUPTI_METRICS -x VT_CUDATRACE_BUFFER_SIZE -tag-output --display-map -am !TBG_dstPath/tbg/openib.conf --mca mpi_leave_pinned 0 -npernode !TBG_gpusPerNode -n !TBG_tasks !TBG_dstPath/input/bin/picongpu !TBG_author !TBG_programParams | tee output
fi

mpiexec --prefix $MPIHOME -x LIBRARY_PATH -npernode !TBG_gpusPerNode -n !TBG_tasks /usr/bin/env bash -c "killall -9 picongpu 2>/dev/null || true"
