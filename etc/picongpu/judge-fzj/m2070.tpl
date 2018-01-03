#!/usr/bin/env bash
# Copyright 2013-2018 Axel Huebl, Rene Widera, Richard Pausch, Wen Fu
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


# PIConGPU batch script for judge moab batch system

#MSUB -q !TBG_queue
#MSUB -l walltime=!TBG_wallTime
#Sets batch job's name
#MSUB -N !TBG_jobName
#MSUB -l nodes=!TBG_nodes:ppn=!TBG_coresPerNode:gpus=!TBG_gpusPerNode:!TBG_gpuType
#MSUB -l mem=20gb
#MSUB -l pmem=10gb
##MSUB -l naccesspolicy=singlejob
# send me mails on job (b)egin, (e)nd, (a)bortion or (n)o mail
#MSUB -m !TBG_mailSettings -M !TBG_mailAddress
#MSUB -d !TBG_dstPath

#MSUB -o stdout
#MSUB -e stderr


## calculation are done by tbg ##
.TBG_gpuType="m2070"
.TBG_queue="largemem"

# settings that can be controlled by environment variables before submit
.TBG_mailSettings=${MY_MAILNOTIFY:-"n"}
.TBG_mailAddress=${MY_MAIL:-"someone@example.com"}
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}
.TBG_profile=${PIC_PROFILE:-"~/picongpu.profile"}

#number of cores per parallel node / default is 2 cores per gpu on k20 queue

# 2 gpus per node if we need more than 2 gpus else same count as TBG_tasks
.TBG_gpusPerNode=`if [ $TBG_tasks -gt 2 ] ; then echo 2; else echo $TBG_tasks; fi`

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
source !TBG_profile
if [ $? -ne 0 ] ; then
  echo "Error: PIConGPU environment profile under \"!TBG_profile\" not found!"
  exit 1
fi

mkdir simOutput 2> /dev/null
cd simOutput

# test if cuda_memtest binary is available
if [ -f !TBG_dstPath/input/bin/cuda_memtest ] ; then
  mpiexec -np !TBG_tasks --mca mpi_leave_pinned 0 !TBG_dstPath/input/bin/cuda_memtest.sh
else
  echo "no binary 'cuda_memtest' available, skip GPU memory test" >&2
fi

if [ $? -eq 0 ] ; then
   mpiexec -np !TBG_tasks --mca mpi_leave_pinned 0 !TBG_dstPath/input/bin/picongpu !TBG_author !TBG_programParams | tee output
fi
