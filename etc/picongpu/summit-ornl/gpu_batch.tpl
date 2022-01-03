#!/usr/bin/env bash
# Copyright 2019-2022 Axel Huebl, Rene Widera
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


# PIConGPU batch script for Summit's LSF (bsub) batch system
#   https://www.olcf.ornl.gov/for-users/system-user-guides/summit/summit-user-guide/#running-jobs

#BSUB -q !TBG_queue
#BSUB -W !TBG_wallTimeNoSeconds
# Sets batch job's name
#BSUB -J !TBG_jobName
#BSUB -nnodes !TBG_nodes
#BSUB -alloc_flags smt4
# send me mails on job (-B)egin, Fi(-N)ish
#BSUB !TBG_mailSettings -u !TBG_mailAddress
#BSUB -cwd !TBG_dstPath
#BSUB -P !TBG_nameProject

#BSUB -o stdout.%J
#BSUB -e stderr.%J


## calculations will be performed by tbg ##
.TBG_queue="batch"

# remove seconds from walltime
.TBG_wallTimeNoSeconds=${TBG_wallTime::-3}

# settings that can be controlled by environment variables before submit
.TBG_mailSettings=${MY_MAILNOTIFY:-""}
.TBG_mailAddress=${MY_MAIL:-"someone@example.com"}
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}
.TBG_nameProject=${proj:-""}
.TBG_profile=${PIC_PROFILE:-"${PROJWORK}/!TBG_nameProject/${USER}/picongpu.profile"}

# number of available/hosted GPUs per node in the system
.TBG_numHostedGPUPerNode=6

# required GPUs per node for the current job
.TBG_gpusPerNode=`if [ $TBG_tasks -gt $TBG_numHostedGPUPerNode ] ; then echo $TBG_numHostedGPUPerNode; else echo $TBG_tasks; fi`

# number of cores to block per GPU - we got 2x22 HW CPU cores per node
#   and we will be accounted those anyway
.TBG_coresPerGPU=7

# use ceil to caculate nodes
.TBG_nodes="$((( TBG_tasks + TBG_gpusPerNode - 1 ) / TBG_gpusPerNode))"

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

# fix MPI collectives by disabling IBM's optimized barriers
# https://github.com/ComputationalRadiationPhysics/picongpu/issues/3814
export OMPI_MCA_coll_ibm_skip_barrier=true

#jsrun  -N 1 -n !TBG_nodes !TBG_dstPath/input/bin/cuda_memtest.sh

# I/O tuning inspired from WarpX, see https://github.com/ECP-WarpX/WarpX/pull/2495
# ROMIO has a hint for GPFS named IBM_largeblock_io which optimizes I/O with operations on large blocks
export IBM_largeblock_io=true

# MPI-I/O: ROMIO hints for parallel HDF5 performance
export OMPI_MCA_io=romio321
export ROMIO_HINTS=./romio-hints
#   number of hosts: unique node names minus batch node
NUM_HOSTS=$(( $(echo $LSB_HOSTS | tr ' ' '\n' | uniq | wc -l) - 1 ))
cat > romio-hints << EOL
romio_cb_write enable
romio_ds_write enable
cb_buffer_size 16777216
cb_nodes ${NUM_HOSTS}
EOL

#if [ $? -eq 0 ] ; then
export OMP_NUM_THREADS=!TBG_coresPerGPU
jsrun --nrs !TBG_tasks --tasks_per_rs 1 --cpu_per_rs !TBG_coresPerGPU --gpu_per_rs 1 --latency_priority GPU-CPU --bind rs --smpiargs="-gpu" !TBG_dstPath/input/bin/picongpu --mpiDirect !TBG_author !TBG_programParams | tee output
# note: instead of the PIConGPU binary, one can also debug starting "js_task_info | sort"
#fi
