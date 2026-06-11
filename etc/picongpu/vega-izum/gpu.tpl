#!/usr/bin/env bash

#
# SPDX-License-Identifier: GPL-3.0-or-later

#

# PIConGPU batch script for Meluxina's SLURM batch system

#SBATCH --partition=!TBG_queue
#SBATCH --time=!TBG_wallTime
# Sets batch job's name
#SBATCH --job-name=!TBG_jobName
#SBATCH --nodes=!TBG_nodes
#SBATCH --ntasks=!TBG_tasks
#SBATCH --ntasks-per-node=!TBG_devicesPerNode
#SBATCH --cpus-per-task=!TBG_coresPerTask
#SBATCH --mincpus=!TBG_mpiTasksPerNode
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:!TBG_devicesPerNode
#SBATCH --mail-type=!TBG_mailSettings
#SBATCH --mail-user=!TBG_mailAddress
#SBATCH --chdir=!TBG_dstPath

#SBATCH -o stdout
#SBATCH -e stderr

## calculations will be performed by tbg ##
.TBG_queue="gpu"

# settings that can be controlled by environment variables before submit
.TBG_mailSettings=${MY_MAILNOTIFY:-"NONE"}
.TBG_mailAddress=${MY_MAIL:-"someone@example.com"}
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}
.TBG_profile=${PIC_PROFILE:-"~/picongpu.profile"}

# number of available/hosted devices per node in the system
.TBG_numHostedDevicesPerNode=4

# required GPUs per node for the current job
.TBG_devicesPerNode=$(if [ $TBG_tasks -gt $TBG_numHostedDevicesPerNode ] ; then echo $TBG_numHostedDevicesPerNode; else echo $TBG_tasks; fi)

# Cores per task. Theoretically we have 128 cores, we might leave one per task for the OS but then we would need to
# hope that srun will do the pinning of cores to memory correctly in order to performantly read from memory.
.TBG_coresPerTask=32

# We only start 1 MPI task per device
.TBG_mpiTasksPerNode="$(( TBG_devicesPerNode * 1 ))"

# use ceil to caculate nodes
.TBG_nodes="$((( TBG_tasks + TBG_devicesPerNode - 1 ) / TBG_devicesPerNode))"

## end calculations ##

echo 'Running program...'

TBG_dstPath="!TBG_dstPath"
cd $TBG_dstPath

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

## Note:
# The Meluxina documentation (https://docs.lxp.lu/first-steps/handling_jobs/) -> "Possible pitfall with --cpus-per-task flag"
# tells us that we need to repeat the --cpus-per-task flag argument in the srun command

# test if cuda_memtest binary is available and we have the node exclusive
if [ -f $TBG_dstPath/input/bin/cuda_memtest ] && [ !TBG_numHostedDevicesPerNode -eq !TBG_devicesPerNode ] ; then
  # Run CUDA memtest to check GPU's health
  srun --cpus-per-task=!TBG_coresPerTask $TBG_dstPath/input/bin/cuda_memtest.sh
else
  echo "Note: GPU memory test was skipped as no binary 'cuda_memtest' available or compute node is not exclusively allocated. This does not affect PIConGPU, starting it now" >&2
fi

if [ $? -eq 0 ] ; then
  # Run PIConGPU
  srun --cpus-per-task=!TBG_coresPerTask -- $TBG_dstPath/input/bin/picongpu !TBG_author !TBG_programParams
fi
