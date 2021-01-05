#!/usr/bin/env bash
# Copyright 2013-2021 Axel Huebl, Richard Pausch, Alexander Matthes
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

#SBATCH -p !TBG_queue
#SBATCH --constraint="Quadrant&Cache"
#SBATCH --time=!TBG_wallTime
# Sets batch job's name
#SBATCH --job-name=!TBG_jobName
#SBATCH -N !TBG_nodes
#SBATCH -c !TBG_coresPerAcc
#SBATCH --mem=90000
#SBATCH --mail-type=!TBG_mailSettings
#SBATCH --mail-user=!TBG_mailAddress
#SBATCH --chdir=!TBG_dstPath

#SBATCH -o stdout
#SBATCH -e stderr


## calculations will be performed by tbg ##
.TBG_queue="knl"

# settings that can be controlled by environment variables before submit
.TBG_mailSettings=${MY_MAILNOTIFY:-"NONE"}
.TBG_mailAddress=${MY_MAIL:-"someone@example.com"}
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}
.TBG_profile=${PIC_PROFILE:-"~/picongpu.profile"}

# number of hardware threads used per core (hyperthreading)
.TBG_hardwareThreadsPerCore=1

# 1 accelerator per node
.TBG_accPerNode=1

# number of cores to block per KNL - we got 64 cores per node
.TBG_coresPerAcc=64

# We only start 1 MPI task per Node
.TBG_mpiTasksPerNode=!TBG_accPerNode

# use ceil to caculate nodes
.TBG_nodes=!TBG_tasks

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

# Run PIConGPU
NUMA_HW_THREADS_PER_PHYSICAL_CORE=!TBG_hardwareThreadsPerCore mpiexec !TBG_dstPath/input/etc/picongpu/cpuNumaStarter.sh !TBG_dstPath/input/bin/picongpu !TBG_author !TBG_programParams
