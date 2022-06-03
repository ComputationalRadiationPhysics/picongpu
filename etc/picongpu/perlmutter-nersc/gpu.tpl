#!/usr/bin/env bash
# Copyright 2021 Axel Huebl
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


# PIConGPU batch script for Perlmutter's SLURM batch system

#SBATCH -p !TBG_queue
#SBATCH --constraint=gpu
#SBATCH --gpus=!TBG_tasks
#SBATCH --time=!TBG_wallTime
#SBATCH --nodes=!TBG_nodes
#SBATCH --ntasks=!TBG_tasks
#SBATCH --cpus-per-task=!TBG_coresPerGPU
#SBATCH --mem=!TBG_memPerNode
#SBATCH --chdir=!TBG_dstPath

# Sets batch job's name
#SBATCH --job-name=!TBG_jobName
#SBATCH --mail-type=!TBG_mailSettings
#SBATCH --mail-user=!TBG_mailAddress
#SBATCH --account=!TBG_nameProject

#SBATCH -o stdout
#SBATCH -e stderr


## calculations will be performed by tbg ##
.TBG_queue="gpu"

# settings that can be controlled by environment variables before submit
.TBG_mailSettings=${MY_MAILNOTIFY:-"NONE"}
.TBG_mailAddress=${MY_MAIL:-"someone@example.com"}
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}
.TBG_nameProject=${proj:-""}"_g"
.TBG_profile=${PIC_PROFILE:-"~/picongpu.profile"}


# number of available/hosted GPUs per node in the system
.TBG_numHostedGPUPerNode=4

# required GPUs per node for the current job
.TBG_gpusPerNode=`if [ $TBG_tasks -gt $TBG_numHostedGPUPerNode ] ; then echo $TBG_numHostedGPUPerNode; else echo $TBG_tasks; fi`

# number of cores to block per A100 - per node, we got 1 Epyc CPU with
# 64 cores a 2 threads per core. We utilise only physical cores
.TBG_coresPerGPU=32

.TBG_totalHostMemory=230000
# host memory per gpu
.TBG_memPerGPU="$(( $TBG_totalHostMemory / $TBG_numHostedGPUPerNode))"
# host memory per node
.TBG_memPerNode="$((TBG_memPerGPU * TBG_gpusPerNode))"

# use ceil to caculate nodes
.TBG_nodes="$((( TBG_tasks + TBG_gpusPerNode - 1 ) / TBG_gpusPerNode))"

## end calculations ##

echo 'Running program...'

cd !TBG_dstPath

# note: no need to source profile as environment is cloned from submit environment
# source !TBG_profile
echo "profile cloned at submit time: !TBG_profile"

#set user rights to u=rwx;g=r-x;o=---
umask 0027

mkdir simOutput 2> /dev/null
cd simOutput
ln -s ../stdout output

# test if cuda_memtest binary is available and we have the node exclusive
if [ -f !TBG_dstPath/input/bin/cuda_memtest ] && [ !TBG_numHostedGPUPerNode -eq !TBG_gpusPerNode ] ; then
  # Run CUDA memtest to check GPU's health
  srun !TBG_dstPath/input/bin/cuda_memtest.sh
else
  echo "Note: GPU memory test was skipped as no binary 'cuda_memtest' available or compute node is not exclusively allocated. This does not affect PIConGPU, starting it now" >&2
fi

export OMP_NUM_THREADS=!TBG_coresPerGPU

# In accordance with the example at
# https://docs.nersc.gov/systems/perlmutter/running-jobs/#4-nodes-16-tasks-16-gpus-1-gpu-visible-to-each-task

export SLURM_CPU_BIND="cores"
if [ $? -eq 0 ] ; then
  # Run PIConGPU
  srun --cpu_bind=cores                 \
       !TBG_dstPath/input/bin/picongpu !TBG_author !TBG_programParams
fi
