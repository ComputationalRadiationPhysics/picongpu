#!/usr/bin/env bash
# Copyright 2013-2018 Axel Huebl, Rene Widera, Robert Dietric
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


# PIConGPU batch script for keenland PBS PRO batch system

#PBS -q !TBG_queue
#PBS -l walltime=!TBG_wallTime
# Sets batch job's name
#PBS -N !TBG_jobName
#PBS -l nodes=!TBG_nodes:ppn=!TBG_coresPerNode:gpus=!TBG_gpusPerNode
# send me mails on job (b)egin, (e)nd, (a)bortion or (n)o mail
#PBS -m !TBG_mailSettings -M !TBG_mailAddress
#PBS -d !TBG_dstPath

#PBS -o stdout
#PBS -e stderr


## calculation are done by tbg ##
.TBG_queue="parallel"

# settings that can be controlled by environment variables before submit
.TBG_mailSettings=${MY_MAILNOTIFY:-"n"}
.TBG_mailAddress=${MY_MAIL:-"someone@example.com"}
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}
.TBG_profile=${PIC_PROFILE:-"~/picongpu.profile"}

# 3 gpus per node if we need more than 3 gpus else same count as TBG_tasks
.TBG_gpusPerNode=`if [ $TBG_tasks -gt 3 ] ; then echo 3; else echo $TBG_tasks; fi`

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

#source /opt/modules-3.2.6/Modules/3.2.6/init/bash

module unload compiler mpi
module rm cuda intel PE-intel openmpi
module add cuda/4.1 gcc/4.4.0 openmpi/1.5.1-gnu autoconf hdf5/1.8.7 pngwriter boost/1.47.0

export MPIHOME=$MPI_ROOT
export SPLASH_ROOT=$HOME/svn_local/splash/src

unset MODULES_NO_OUTPUT


mkdir simOutput 2> /dev/null
cd simOutput

# test if cuda_memtest binary is available
if [ -f !TBG_dstPath/input/bin/cuda_memtest ] ; then
  mpiexec  --mca btl openib,self,sm --mca mpi_leave_pinned 0 !TBG_dstPath/input/bin/cuda_memtest.sh
else
  echo "no binary 'cuda_memtest' available, skip GPU memory test" >&2
fi

if [ $? -eq 0 ] ; then
   mpiexec  --display-map --mca btl openib,self,sm --display-map -am !TBG_dstPath/tbg/openib.conf --mca mpi_leave_pinned 0 !TBG_dstPath/input/bin/picongpu !TBG_author !TBG_programParams | tee output
fi

