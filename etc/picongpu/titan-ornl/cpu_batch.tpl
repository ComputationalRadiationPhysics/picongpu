#!/usr/bin/env bash
# Copyright 2013-2019 Axel Huebl, Rene Widera
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


# PIConGPU batch script for titan PBS batch system

#PBS -q !TBG_queue
#PBS -l walltime=!TBG_wallTime
# Sets batch job's name
#PBS -N !TBG_jobName
#PBS -l nodes=!TBG_nodes
# send me mails on job (b)egin, (e)nd, (a)bortion or (n)o mail
#PBS -m !TBG_mailSettings -M !TBG_mailAddress
#PBS -d !TBG_dstPath
#PBS -A !TBG_nameProject

#PBS -o stdout
#PBS -e stderr

#PBS -l gres=atlas1%atlas2


## calculations will be performed by tbg ##
.TBG_queue="batch"

# settings that can be controlled by environment variables before submit
.TBG_mailSettings=${MY_MAILNOTIFY:-"n"}
.TBG_mailAddress=${MY_MAIL:-"someone@example.com"}
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}
.TBG_nameProject=${proj:-""}
.TBG_profile=${PIC_PROFILE:-"${PROJWORK}/!TBG_nameProject/${USER}/picongpu.profile"}

# 2 packages per node
.TBG_devicesPerNode=2

# Cores (processing units) per Compute Unit (CU) regarding int & float units
# 2: allowing contention for the compute unit's floating point scheduler
# 1: otherwise if float ops dominate
.TBG_corePerCU=2

# number of cores to block per device (package)
# allowing contention for the compute unit's floating point scheduler
.TBG_coresPerDevice=$(( 4 * TBG_corePerCU ))

# use ceil to caculate nodes
.TBG_nodes=$(( ( TBG_tasks + TBG_devicesPerNode - 1 ) / TBG_devicesPerNode))

## end calculations ##

echo 'Running program...'
echo !TBG_jobName

cd !TBG_dstPath
echo -n "present working directory:"
pwd

source !TBG_profile 2>/dev/null
if [ $? -ne 0 ] ; then
  echo "Error: PIConGPU environment profile under \"!TBG_profile\" not found!"
  exit 1
fi

mkdir simOutput 2> /dev/null
cd simOutput

export OMP_NUM_THREADS=!TBG_coresPerDevice

# todo: pinning not yet verified, please check!
# PE: Processing Element, basically an Unix 'process' which is an MPI task
# -N, --pes-per-node: PEs per node
# -S, --pes-per-numa-node: PEs per NUMA node (before we start OMP threads)
# -j, --cpus-per-cu: CPUs to use per Compute Unit (CU) regarding int & float units
# -d, --cpus-per-pe depth: Number of CPUs allocated per PE (number of threads)
# --ss, --strict-memory-containment: Strict memory containment per NUMA node
aprun -N 2 -S 1 -j !TBG_corePerCU -d !TBG_coresPerDevice --ss \
    -n !TBG_tasks !TBG_dstPath/input/bin/picongpu !TBG_author !TBG_programParams | tee output
