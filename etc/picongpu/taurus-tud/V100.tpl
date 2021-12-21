#!/usr/bin/env bash
# Copyright 2013-2021 Axel Huebl, Richard Pausch, Alexander Debus, Klaus Steiniger
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


# PIConGPU batch script for taurus' SLURM batch system

#SBATCH --partition=!TBG_queue
#SBATCH --time=!TBG_wallTime
# Sets batch job's name
#SBATCH --job-name=!TBG_jobName
#SBATCH --nodes=!TBG_nodes
#SBATCH --ntasks=!TBG_tasks
#SBATCH --ntasks-per-node=!TBG_gpusPerNode
#SBATCH --mincpus=!TBG_mpiTasksPerNode
#SBATCH --cpus-per-task=!TBG_coresPerGPU
#SBATCH --mem=0
#SBATCH --gres=gpu:!TBG_gpusPerNode
#SBATCH --exclusive

# disable hyperthreading (default on taurus)
#SBATCH --hint=nomultithread

#SBATCH --mail-type=!TBG_mailSettings
#SBATCH --mail-user=!TBG_mailAddress
#SBATCH --chdir=!TBG_dstPath

#SBATCH -o stdout
#SBATCH -e stderr


## calculations will be performed by tbg ##
.TBG_queue="ml"

# settings that can be controlled by environment variables before submit
.TBG_mailSettings=${MY_MAILNOTIFY:-"ALL"}
.TBG_mailAddress=${MY_MAIL:-"someone@example.com"}
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}
.TBG_profile=${PIC_PROFILE:-"~/picongpu.profile"}

# 6 gpus per node
# Taurus does not have enough node memory to hold data of all GPUs in node memory during ADIOS output.
# If you experience crashes with memory allocation errors or get killed by the batch system's
# resource watch dog, reduce the number of GPUs used per node to three here for debugging.
# That is, replace in the following line the two appearances of 6 with 3.
.TBG_gpusPerNode=`if [ $TBG_tasks -gt 6 ] ; then echo 6; else echo $TBG_tasks; fi`

# number of CPU cores to block per GPU
# we got 7 CPU cores per GPU (44cores/6gpus ~ 7cores)
.TBG_coresPerGPU=7

# We only start 1 MPI task per GPU
.TBG_mpiTasksPerNode="$(( TBG_gpusPerNode * 1 ))"

# use ceil to calculate nodes
.TBG_nodes="$((( TBG_tasks + TBG_gpusPerNode -1 ) / TBG_gpusPerNode))"

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

# set user rights to u=rwx;g=r-x;o=---
umask 0027

mkdir simOutput 2> /dev/null
cd simOutput
ln -s ../stdout output

# we are not sure if the current bullxmpi/1.2.4.3 catches pinned memory correctly
#   support ticket [Ticket:2014052241001186] srun: mpi mca flags
#   see bug https://github.com/ComputationalRadiationPhysics/picongpu/pull/438
export OMPI_MCA_mpi_leave_pinned=0

# The OMPIO backend in OpenMPI up to 3.1.3 and 4.0.0 is broken, use the
# fallback ROMIO backend instead.
#   see bug https://github.com/open-mpi/ompi/issues/6285
export OMPI_MCA_io=^ompio

# test if cuda_memtest binary is available
if [ -f !TBG_dstPath/input/bin/cuda_memtest ] ; then
  # Run CUDA memtest to check GPU's health
  srun -K1 !TBG_dstPath/input/bin/cuda_memtest.sh
else
  echo "Note: GPU memory test was skipped as no binary 'cuda_memtest' available. This does not affect PIConGPU, starting it now" >&2
fi

if [ $? -eq 0 ] ; then
  # Run PIConGPU
  srun -K1 !TBG_dstPath/input/bin/picongpu !TBG_author !TBG_programParams
fi

