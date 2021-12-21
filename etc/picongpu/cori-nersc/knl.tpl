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


# PIConGPU batch script for Cori's SLURM batch system

#SBATCH -p !TBG_queue
#SBATCH --constraint="knl,quad,cache"
#SBATCH --time=!TBG_wallTime
#SBATCH --nodes=!TBG_nodes
#SBATCH --core-spec=!TBG_coresForSystem
#SBATCH --chdir=!TBG_dstPath

# Sets batch job's name
#SBATCH --job-name=!TBG_jobName
#SBATCH --mail-type=!TBG_mailSettings
#SBATCH --mail-user=!TBG_mailAddress
#SBATCH --account=!TBG_nameProject

#SBATCH -o stdout
#SBATCH -e stderr


## calculations will be performed by tbg ##
.TBG_queue="regular"

# settings that can be controlled by environment variables before submit
.TBG_mailSettings=${MY_MAILNOTIFY:-"NONE"}
.TBG_mailAddress=${MY_MAIL:-"someone@example.com"}
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}
.TBG_nameProject=${proj:-""}
.TBG_profile=${PIC_PROFILE:-"~/picongpu.profile"}

# number of hardware threads used per core (hyperthreading: 1-4)
.TBG_hardwareThreadsPerCore=1

# 1 accelerator per node
.TBG_accPerNode=1

# number of cores to block per KNL - we got 64(+4) cores per node
.TBG_coresPerAcc=64
.TBG_coresForSystem=4

# use ceil to calculate nodes
.TBG_nodes=!TBG_tasks

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

export OMP_NUM_THREADS=$(( !TBG_coresPerAcc * !TBG_hardwareThreadsPerCore ))
export OMP_PLACES=threads
export OMP_PROC_BIND=spread  # variants: spread and close

# Run PIConGPU
srun --cpu_bind=cores                 \
     --ntasks=!TBG_tasks              \
     --cpus-per-task=!TBG_coresPerAcc \
     !TBG_dstPath/input/bin/picongpu !TBG_author !TBG_programParams
