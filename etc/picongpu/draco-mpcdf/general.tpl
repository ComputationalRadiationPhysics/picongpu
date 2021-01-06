#!/usr/bin/env bash
# Copyright 2013-2021 Axel Huebl, Richard Pausch, Rene Widera
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
#SBATCH --mincpus=!TBG_mpiTasksPerNode
#SBATCH --cpus-per-task=!TBG_coresPerGPU
#SBATCH --exclusive
#SBATCH --mail-type=!TBG_mailSettings
#SBATCH --mail-user=!TBG_mailAddress
#SBATCH --workdir=!TBG_dstPath

#SBATCH -o stdout
#SBATCH -e stderr


## calculations will be performed by tbg ##
.TBG_queue="general"

# settings that can be controlled by environment variables before submit
.TBG_mailSettings=${MY_MAILNOTIFY:-"NONE"}
.TBG_mailAddress=${MY_MAIL:-"someone@example.com"}
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}
.TBG_profile=${PIC_PROFILE:-"~/picongpu.profile"}

# 1 gpus per node
.TBG_gpusPerNode=2

# number of cores to block per GPU - we got 6 cpus per gpu
#   and we will be accounted 6 CPUs per GPU anyway
.TBG_coresPerGPU=32

# We only start 2 MPI task per GPU
.TBG_mpiTasksPerNode="!TBG_gpusPerNode"

# use ceil to caculate nodes
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

#set user rights to u=rwx;g=r-x;o=---
umask 0027

mkdir simOutput 2> /dev/null
cd simOutput
ln -s ../stdout output

#wait that all nodes see ouput folder
sleep 1

# Run PIConGPU
srun -K1 !TBG_dstPath/tbg/cpuNumaStarter.sh !TBG_dstPath/input/bin/picongpu !TBG_author !TBG_programParams


