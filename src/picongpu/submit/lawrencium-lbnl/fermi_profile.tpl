#!/usr/bin/env bash
# Copyright 2013-2014 Axel Huebl
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

TBG_queue="mako_manycore"
TBG_account="ac_blast"
TBG_qos="mako_normal"
TBG_feature="mako_fermi"

TBG_mailSettings="ALL"
TBG_mailAdress="someone@example.com"

# 2 gpus per node
TBG_gpusPerNode=`if [ $TBG_tasks -gt 2 ] ; then echo 2; else echo $TBG_tasks; fi`

# number of cores to block per GPU - we use one right now
TBG_coresPerGPU=1

# We only start 1 MPI task per GPU
TBG_mpiTasksPerNode="$(( TBG_gpusPerNode * 1 ))"

# use ceil to calculate the number of nodes
TBG_nodes="$(( ( TBG_tasks + TBG_gpusPerNode -1 ) / TBG_gpusPerNode))"

# in MB; 24 GB per node on 12 CPUs
TBG_memPerCPU="$(( 24000 / TBG_gpusPerNode ))M"

## end calculations ##

# PIConGPU batch script for LBL lawrencium's SLURM batch system
#   https://sites.google.com/a/lbl.gov/high-performance-computing-services-group/lbnl-supercluster/lawrencium

#SBATCH --partition=!TBG_queue
#SBATCH --time=!TBG_wallTime
# Sets batch job's name
#SBATCH --job-name=!TBG_jobName
#SBATCH --nodes=!TBG_nodes
#SBATCH --ntasks-per-node=!TBG_mpiTasksPerNode
#SBATCH --cpus-per-task=!TBG_coresPerGPU
#SBATCH --mem-per-cpu=!TBG_memPerCPU
#SBATCH --constraint=!TBG_feature
#SBATCH --qos=!TBG_qos
# send me a mail on (b)egin, (e)nd, (a)bortion
#SBATCH --mail-type=!TBG_mailSettings
#SBATCH --mail-user=!TBG_mailAdress
#SBATCH --workdir=!TBG_dstPath
#SBATCH --account=!TBG_account

#SBATCH -o stdout
#SBATCH -e stderr

echo 'Running program...'

cd !TBG_dstPath

export MODULES_NO_OUTPUT=1
source ~/picongpu.profile
unset MODULES_NO_OUTPUT

#set user rights to u=rwx;g=r-x;o=---
umask 0027

mkdir simOutput 2> /dev/null
cd simOutput

# openmpi/1.6.5 is not GPU aware and handles pinned memory correctly incorrectly
#   see bug https://github.com/ComputationalRadiationPhysics/picongpu/pull/438
export OMPI_MCA_mpi_leave_pinned=0

# Run CUDA memtest to check GPU's health
mpirun !TBG_dstPath/picongpu/bin/cuda_memtest.sh

# Run PIConGPU
if [ $? -eq 0 ] ; then
  mpirun !TBG_dstPath/picongpu/bin/picongpu !TBG_programParams
fi
