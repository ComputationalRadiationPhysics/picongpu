#!/usr/bin/env bash
# Copyright 2013-2023 Axel Huebl, Richard Pausch, Rene Widera,
#                     Marco Garten, Alexander Debus,
#                     Jian Fuh Ong, Andrei Berceanu
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


# PIConGPU batch script for ELI-NP's SLURM batch system

#SBATCH --account=!TBG_account
#SBATCH --partition=!TBG_queue
#SBATCH --time=!TBG_wallTime
# Sets batch job's name
#SBATCH --job-name=!TBG_jobName
#SBATCH --nodes=!TBG_nodes
#SBATCH --ntasks=!TBG_tasks
#SBATCH --ntasks-per-node=!TBG_gpusPerNode
#SBATCH --mincpus=!TBG_mpiTasksPerNode
#SBATCH --cpus-per-task=!TBG_coresPerGPU
#SBATCH --gres=gpu:!TBG_gpusPerNode
#SBATCH --mem=!TBG_memPerNode
#SBATCH --exclusive

#SBATCH --mail-type=!TBG_mailSettings
#SBATCH --mail-user=!TBG_mailAddress
#SBATCH --chdir=!TBG_dstPath

#SBATCH -o stdout
#SBATCH -e stderr


## calculations will be performed by tbg ##
.TBG_queue=${TBG_partition:-"gpu"}
.TBG_account=${USER}_a+

# settings that can be controlled by environment variables before submit
.TBG_mailSettings=${MY_MAILNOTIFY:-"NONE"}
.TBG_mailAddress=${MY_MAIL:-"someone@example.com"}
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}
.TBG_profile=${PIC_PROFILE:-"~/gpu_v100_picongpu.profile"}

# number of available/hosted GPUs per node in the system
.TBG_numHostedGPUPerNode=16

# required GPUs per node for the current job
.TBG_gpusPerNode=$(if [ $TBG_tasks -gt $TBG_numHostedGPUPerNode ] ; then echo $TBG_numHostedGPUPerNode; else echo $TBG_tasks; fi)

# host memory per gpu
.TBG_memPerGPU="$((1300000 / $TBG_numHostedGPUPerNode))"
# host memory per node
.TBG_memPerNode="$((TBG_memPerGPU * TBG_gpusPerNode))"

# number of cores to block per GPU
.TBG_coresPerGPU=3

# We only start 1 MPI task per GPU
.TBG_mpiTasksPerNode="$(( TBG_gpusPerNode * 1 ))"

# use ceil to calculate nodes
.TBG_nodes="$((( TBG_tasks + TBG_gpusPerNode - 1 ) / TBG_gpusPerNode))"

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

# test if cuda_memtest binary is available
if [ -f !TBG_dstPath/input/bin/cuda_memtest ] ; then
  # Run CUDA memtest to check GPU's health
  mpirun -n !TBG_tasks !TBG_dstPath/input/bin/cuda_memtest.sh
else
  echo "Note: GPU memory test was skipped as no binary 'cuda_memtest' available. This does not affect PIConGPU, starting it now" >&2
fi

if [ $? -eq 0 ] ; then
  # Run PIConGPU
  mpirun -n !TBG_tasks !TBG_dstPath/input/bin/picongpu !TBG_author !TBG_programParams
fi
