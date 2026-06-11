#!/usr/bin/env bash

#
# SPDX-License-Identifier: GPL-3.0-or-later

#

# PIConGPU batch script for Discoverer's SLURM batch system

#SBATCH --account=!TBG_account
#SBATCH --qos=!TBG_qos
#SBATCH --partition=!TBG_partition
#SBATCH --time=!TBG_wallTime
# Sets batch job's name
#SBATCH --job-name=!TBG_jobName
#SBATCH --nodes=!TBG_nodes
#SBATCH --gres=gpu:!TBG_devicesPerNode
#SBATCH --ntasks=!TBG_tasks
#SBATCH --ntasks-per-node=!TBG_devicesPerNode
#SBATCH --cpus-per-task=!TBG_coresPerGPU
#SBATCH --hint=nomultithread
#SBATCH --mincpus=!TBG_mpiTasksPerNode
#SBATCH --mem=!TBG_memPerNode
#SBATCH --mail-type=!TBG_mailSettings
#SBATCH --mail-user=!TBG_mailAddress
#SBATCH --chdir=!TBG_dstPath

#SBATCH -o stdout
#SBATCH -e stderr

## calculations will be performed by tbg ##

# settings that can be controlled by environment variables before submit
.TBG_mailSettings=${MY_MAILNOTIFY:-"NONE"}
.TBG_mailAddress=${MY_MAIL:-"someone@example.com"}
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}
.TBG_account=${account:-""}
.TBG_qos=${qos:-""}
.TBG_partition=${disco_partition:-"common"}
.TBG_profile=${PIC_PROFILE:-"~/picongpu.profile"}

# number of available/hosted devices per node in the system
.TBG_numHostedDevicesPerNode=8

# required GPUs per node for the current job
.TBG_devicesPerNode=$(if [ $TBG_tasks -gt $TBG_numHostedDevicesPerNode ] ; then echo $TBG_numHostedDevicesPerNode; else echo $TBG_tasks; fi)

# host memory per device
.TBG_memPerDevice=255000
# host memory per node
.TBG_memPerNode="$((TBG_memPerDevice * TBG_devicesPerNode))"

# We only start 1 MPI task per device
.TBG_mpiTasksPerNode="$(( TBG_devicesPerNode * 1 ))"

# use ceil to caculate nodes
.TBG_nodes="$((( TBG_tasks + TBG_devicesPerNode - 1 ) / TBG_devicesPerNode))"

## end calculations ##

# according to (https://docs.discoverer.bg/writing_slurm_batch.html#common-resource-allocators-in-slurm-batch-scripts)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=false
export UCX_NET_DEVICES=mlx5_0:1

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

# number of cores to block per GPU
.TBG_coresPerGPU=13

#set user rights to u=rwx;g=r-x;o=---
umask 0027

mkdir simOutput 2> /dev/null
cd simOutput
ln -s ../stdout output

# test if cuda_memtest binary is available and we have the node exclusive
if [ -f $TBG_dstPath/input/bin/cuda_memtest ] && [ !TBG_numHostedDevicesPerNode -eq !TBG_devicesPerNode ] ; then
  # Run CUDA memtest to check GPU's health
  mpirun $TBG_dstPath/input/bin/cuda_memtest.sh
else
  echo "Note: GPU memory test was skipped as no binary 'cuda_memtest' available or compute node is not exclusively allocated. This does not affect PIConGPU, starting it now" >&2
fi

# Run PIConGPU
mpirun $TBG_dstPath/input/bin/picongpu !TBG_author !TBG_programParams
