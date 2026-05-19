#!/usr/bin/env bash
# Copyright 2013-2024 Axel Huebl, Richard Pausch, Rene Widera, Sergei Bastrakov, Klaus Steiniger
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


# PIConGPU batch script for Leonardo's SLURM batch system
# For an example script see (https://docs.hpc.cineca.it/hpc/hpc_scheduler.html#how-to-prepare-a-script-to-submit-jobs)

#SBATCH --account=!TBG_nameProject
#SBATCH --partition=!TBG_queue
#SBATCH --qos=!TBG_qos
#SBATCH --time=!TBG_wallTime
# Sets batch job's name
#SBATCH --job-name=!TBG_jobName
#SBATCH --nodes=!TBG_nodes
#SBATCH --ntasks-per-node=!TBG_devicesPerNode
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=!TBG_coresPerTask
#SBATCH --mincpus=!TBG_mpiTasksPerNode
#SBATCH --hint=nomultithread
#SBATCH --mail-type=!TBG_mailSettings
#SBATCH --mail-user=!TBG_mailAddress
#SBATCH --chdir=!TBG_dstPath

#SBATCH -o stdout
#SBATCH -e stderr

## calculations will be performed by tbg ##
## Depending on jobsize, different qos need to be used!
## See (https://docs.hpc.cineca.it/hpc/leonardo.html#job-managing-and-slurm-partitions)
.TBG_queue="boost_usr_prod"

# settings that can be controlled by environment variables before submit
.TBG_mailSettings=${MY_MAILNOTIFY:-"NONE"}
.TBG_mailAddress=${MY_MAIL:-"someone@example.com"}
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}
.TBG_nameProject=${account:-""}
.TBG_qos=${qos:-""}
.TBG_profile=${PIC_PROFILE:-"~/picongpu.profile"}

# number of available/hosted devices per node in the system
.TBG_numHostedDevicesPerNode=4

# required GPUs per node for the current job
.TBG_devicesPerNode=$(if [ $TBG_tasks -gt $TBG_numHostedDevicesPerNode ] ; then echo $TBG_numHostedDevicesPerNode; else echo $TBG_tasks; fi)

# Cores per task. Theoretically we have 48 cores, we might leave one per task for the OS but then we would need to
# hope that srun will do the pinning of cores to memory correctly in order to performantly read from memory.
.TBG_coresPerTask=8

# We only start 1 MPI task per device
.TBG_mpiTasksPerNode="$(( TBG_devicesPerNode * 1 ))"

# use ceil to caculate nodes
.TBG_nodes="$((( TBG_tasks + TBG_devicesPerNode - 1 ) / TBG_devicesPerNode))"

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

# Optional: Set environment variables for performance tuning
export OMP_NUM_THREADS=!TBG_coresPerTask   # Set OpenMP threads per task
export NCCL_DEBUG=INFO                     # Enable NCCL debugging (for multi-GPU communication)

## Note:
# The Meluxina documentation (https://docs.lxp.lu/first-steps/handling_jobs/) -> "Possible pitfall with --cpus-per-task flag"
# tells us that we need to repeat the  --cpus-per-task flag argument in the srun command

# test if cuda_memtest binary is available and we have the node exclusive
if [ -f !TBG_dstPath/input/bin/cuda_memtest ] && [ !TBG_numHostedDevicesPerNode -eq !TBG_devicesPerNode ] ; then
  # Run CUDA memtest to check GPU's health
  srun --cpus-per-task=!TBG_coresPerTask !TBG_dstPath/input/bin/cuda_memtest.sh
else
  echo "Note: GPU memory test was skipped as no binary 'cuda_memtest' available or compute node is not exclusively allocated. This does not affect PIConGPU, starting it now" >&2
fi

if [ $? -eq 0 ] ; then
  # Run PIConGPU
  srun --cpus-per-task=!TBG_coresPerTask -- !TBG_dstPath/input/bin/picongpu !TBG_author !TBG_programParams
fi
