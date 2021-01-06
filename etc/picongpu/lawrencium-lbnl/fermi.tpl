#!/usr/bin/env bash
# Copyright 2013-2021 Axel Huebl
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


# PIConGPU batch script for LBL lawrencium's SLURM batch system
#   https://sites.google.com/a/lbl.gov/high-performance-computing-services-group/lbnl-supercluster/lawrencium

#SBATCH --partition=!TBG_queue
#SBATCH --time=!TBG_wallTime
# Sets batch job's name
#SBATCH --job-name=!TBG_jobName
#SBATCH --nodes=!TBG_nodes
#SBATCH --ntasks-per-node=!TBG_mpiTasksPerNode
#SBATCH --cpus-per-task=!TBG_coresPerGPU
#SBATCH --mem-per-cpu=!TBG_memPerCPU
#SBATCH --constraint=!TBG_feature
#SBATCH --qos=!TBG_qos
# send me mails on BEGIN, END, FAIL, REQUEUE, ALL,
# TIME_LIMIT, TIME_LIMIT_90, TIME_LIMIT_80 and/or TIME_LIMIT_50
#SBATCH --mail-type=!TBG_mailSettings
#SBATCH --mail-user=!TBG_mailAddress
#SBATCH --workdir=!TBG_dstPath
#SBATCH --account=!TBG_account

#SBATCH -o stdout
#SBATCH -e stderr


## calculations will be performed by tbg ##

.TBG_queue="mako_manycore"
.TBG_account="ac_blast"
.TBG_qos="mako_normal"
.TBG_feature="mako_fermi"

# settings that can be controlled by environment variables before submit
.TBG_mailSettings=${MY_MAILNOTIFY:-"NONE"}
.TBG_mailAddress=${MY_MAIL:-"someone@example.com"}
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}
.TBG_profile=${PIC_PROFILE:-"~/picongpu.profile"}

# 2 gpus per node
.TBG_gpusPerNode=`if [ $TBG_tasks -gt 2 ] ; then echo 2; else echo $TBG_tasks; fi`

# number of cores to block per GPU - we use one right now
.TBG_coresPerGPU=1

# We only start 1 MPI task per GPU
.TBG_mpiTasksPerNode="$(( TBG_gpusPerNode * 1 ))"

# use ceil to calculate the number of nodes
.TBG_nodes="$(( ( TBG_tasks + TBG_gpusPerNode -1 ) / TBG_gpusPerNode))"

# in MB; 24 GB per node on 12 CPUs
.TBG_memPerCPU="$(( 24000 / TBG_gpusPerNode ))M"

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
ln -s ../stdout output

# openmpi/1.6.5 is not GPU aware and handles pinned memory correctly incorrectly
#   see bug https://github.com/ComputationalRadiationPhysics/picongpu/pull/438
export OMPI_MCA_mpi_leave_pinned=0

# The OMPIO backend in OpenMPI up to 3.1.3 and 4.0.0 is broken, use the
# fallback ROMIO backend instead.
#   see bug https://github.com/open-mpi/ompi/issues/6285
export OMPI_MCA_io=^ompio

# test if cuda_memtest binary is available
if [ -f !TBG_dstPath/input/bin/cuda_memtest ] ; then
  # Run CUDA memtest to check GPU's health
  mpirun !TBG_dstPath/input/bin/cuda_memtest.sh
else
  echo "Note: GPU memory test was skipped as no binary 'cuda_memtest' available. This does not affect PIConGPU, starting it now" >&2
fi

if [ $? -eq 0 ] ; then
  # Run PIConGPU
  mpirun !TBG_dstPath/input/bin/picongpu !TBG_author !TBG_programParams
fi
