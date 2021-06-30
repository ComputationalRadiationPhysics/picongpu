#!/usr/bin/env bash
# Copyright 2013-2021 Axel Huebl, Richard Pausch, Rene Widera,
#                     Marco Garten, Alexander Debus
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


# PIConGPU batch script for perlmutter's SLURM batch system

#SBATCH --partition=!TBG_queue
#SBATCH --account=!TBG_nameProject
#SBATCH --time=!TBG_wallTime
# Sets batch job's name
#SBATCH --job-name=!TBG_jobName
#SBATCH --nodes=!TBG_nodes
#SBATCH --ntasks=!TBG_tasks
#SBATCH --ntasks-per-node=!TBG_gpusPerNode
#SBATCH --cpus-per-task=!TBG_coresPerGPU
#SBATCH --cpu-bind=cores
#SBATCH --constraint="gpu"
# total number gpus is equal to number of tasks in the job
#SBATCH --gpus=!TBG_tasks
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH --mem=!TBG_memPerNode
#SBATCH --mail-type=!TBG_mailSettings
#SBATCH --mail-user=!TBG_mailAddress
#SBATCH --chdir=!TBG_dstPath
# notify the job 240 sec before the wall time ends
##SBATCH --signal=B:SIGALRM@240

#SBATCH -o stdout
#SBATCH -e stderr


## calculations will be performed by tbg ##
.TBG_queue="regular"
.TBG_nameProject=${proj:-""}

# settings that can be controlled by environment variables before submit
.TBG_mailSettings=${MY_MAILNOTIFY:-"NONE"}
.TBG_mailAddress=${MY_MAIL:-"someone@example.com"}
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}
.TBG_profile=${PIC_PROFILE:-"~/picongpu.profile"}

# number of available/hosted GPUs per node in the system
.TBG_numHostedGPUPerNode=4

# required GPUs per node for the current job
.TBG_gpusPerNode=`if [ $TBG_tasks -gt $TBG_numHostedGPUPerNode ] ; then echo $TBG_numHostedGPUPerNode; else echo $TBG_tasks; fi`

# host memory per gpu
.TBG_memPerGPU="$((254000 / $TBG_gpusPerNode))"
# host memory per node
.TBG_memPerNode="$((TBG_memPerGPU * TBG_gpusPerNode))"

# number of cores to block per GPU - we got 32 cpus per gpu
# EPYC 7763 with 128 cores
.TBG_coresPerGPU=32

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

mkdir simOutput 2> /dev/null
cd simOutput
ln -s ../stdout output

# test if cuda_memtest binary is available and we have the node exclusive
if [ -f !TBG_dstPath/input/bin/cuda_memtest ] && [ !TBG_numHostedGPUPerNode -eq !TBG_gpusPerNode ] ; then
  # Run CUDA memtest to check GPU's health
  mpiexec !TBG_dstPath/input/bin/cuda_memtest.sh
else
  echo "Note: GPU memory test was skipped as no binary 'cuda_memtest' available or compute node is not exclusively allocated. This does not affect PIConGPU, starting it now" >&2
fi

if [ $? -eq 0 ] ; then
  export OMP_NUM_THREADS=!TBG_coresPerGPU
  # Run PIConGPU
  source !TBG_dstPath/tbg/handleSlurmSignals.sh srun !TBG_dstPath/input/bin/picongpu --mpiDirect !TBG_author !TBG_programParams
fi
