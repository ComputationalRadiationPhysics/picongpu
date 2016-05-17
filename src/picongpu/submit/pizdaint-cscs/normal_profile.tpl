#!/bin/bash -l
# Copyright 2013-2016 Axel Huebl, Richard Pausch, Rene Widera
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

## calculations will be performed by tbg ##
TBG_queue="normal"

# settings that can be controlled by environment variables before submit
TBG_mailSettings=${MY_MAILNOTIFY:-"ALL"}
TBG_mailAddress=${MY_MAIL:-"someone@example.com"}
TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}

# 1 gpus per node
TBG_gpusPerNode=1

# number of cores to block per GPU - we got 8 cpus per gpu
#   and we will be accounted 8 CPUs per GPU anyway
TBG_coresPerGPU=8

# We only start 1 MPI task per GPU
TBG_mpiTasksPerNode=1

# use ceil to caculate nodes
TBG_nodes=!TBG_tasks

## end calculations ##

# PIConGPU batch script for pizdaint SLURM batch system

#SBATCH --partition=!TBG_queue
#SBATCH --time=!TBG_wallTime
# Sets batch job's name
#SBATCH --job-name=!TBG_jobName
#SBATCH --nodes=!TBG_nodes
#SBATCH --ntasks-per-node=!TBG_coresPerGPU
#SBATCH --ntasks-per-core=1
# send me mails on BEGIN, END, FAIL, REQUEUE, ALL,
# TIME_LIMIT, TIME_LIMIT_90, TIME_LIMIT_80 and/or TIME_LIMIT_50
#SBATCH --mail-type=!TBG_mailSettings
#SBATCH --mail-user=!TBG_mailAddress

#SBATCH -o stdout
#SBATCH -e stderr

echo 'Running program...'

cd !TBG_dstPath

export MODULES_NO_OUTPUT=1
source $SCRATCH/picongpu.profile
unset MODULES_NO_OUTPUT

mkdir simOutput 2> /dev/null
cd simOutput

# Run PIConGPU
aprun  -N 1 -n !TBG_tasks !TBG_dstPath/picongpu/bin/picongpu !TBG_author !TBG_programParams
