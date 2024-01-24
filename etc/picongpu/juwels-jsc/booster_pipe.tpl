#!/usr/bin/env bash
# Copyright 2013-2023 Axel Huebl, Richard Pausch, Rene Widera, Sergei Bastrakov
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


# PIConGPU batch script for JUWELS's SLURM batch system

#SBATCH --account=!TBG_nameProject
#SBATCH --partition=!TBG_queue
#SBATCH --time=!TBG_wallTime
# Sets batch job's name
#SBATCH --job-name=!TBG_jobName
#SBATCH --nodes=!TBG_nodes
#SBATCH --exclusive
#SBATCH --gres=gpu:!TBG_devicesPerNode
#SBATCH --ntasks=!TBG_tasks_total
#SBATCH --ntasks-per-node=!TBG_mpiTasksPerNode
#SBATCH --mem=0 # all of it

##SBATCH --mincpus=!TBG_mpiTasksPerNode
#SBATCH --mail-type=!TBG_mailSettings
#SBATCH --mail-user=!TBG_mailAddress
#SBATCH --chdir=!TBG_dstPath

#SBATCH -o stdout
#SBATCH -e stderr

# Workaround for Infinibands' "Transport retry count exceeded"-error:
# https://apps.fz-juelich.de/jsc/hps/juwels/faq.html#my-job-failed-with-transport-retry-count-exceeded
export UCX_RC_TIMEOUT=3000000.00us # 3s instead of 1s

## calculations will be performed by tbg ##
.TBG_queue="booster"

# settings that can be controlled by environment variables before submit
.TBG_mailSettings=${MY_MAILNOTIFY:-"NONE"}
.TBG_mailAddress=${MY_MAIL:-"someone@example.com"}
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}
.TBG_nameProject=${proj:-""}
.TBG_profile=${PIC_PROFILE:-"~/picongpu.profile"}

# number of available/hosted devices per node in the system
.TBG_numHostedDevicesPerNode=4

# required GPUs per node for the current job
.TBG_devicesPerNode=$(if [ $TBG_tasks -gt $TBG_numHostedDevicesPerNode ] ; then echo $TBG_numHostedDevicesPerNode; else echo $TBG_tasks; fi)

# host memory per device
.TBG_memPerDevice="$((499712 / $TBG_numHostedDevicesPerNode))"
# host memory per node
.TBG_memPerNode="$((TBG_memPerDevice * TBG_devicesPerNode))"

# We only start 1 MPI task per device
.TBG_mpiTasksPerNode="$(( TBG_devicesPerNode * 1 + 1 ))"

# use ceil to caculate nodes
.TBG_nodes="$((( TBG_tasks + TBG_devicesPerNode - 1 ) / TBG_devicesPerNode))"

.TBG_tasks_total="$((TBG_tasks + TBG_nodes))"

.TBG_DataTransport=ucx

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

# more should be possible
.TBG_coresPerGPU=6
.TBG_coresPerPipeInstance=6

# Assign one OpenMP thread per available core per GPU (=task)
export OMP_NUM_THREADS=!TBG_coresPerGPU

#set user rights to u=rwx;g=r-x;o=---
umask 0027

mkdir simOutput 2> /dev/null
cd simOutput
ln -s ../stdout output

# cuda_memtest omitted since GPU mapping is achieved via CUDA_VISIBLE_DEVICES on the Booster
# and cuda_memtest fails in this case.

# Run PIConGPU
export OPENPMD_CHUNK_DISTRIBUTION=slicedataset
mkdir -p openPMD

echo '!TBG_inconfig_pipe' | tee inconfig.json
echo '!TBG_outconfig_pipe' | tee outconfig.json

srun                                          \
  --exclusive                                 \
  --ntasks !TBG_nodes                         \
  --nodes !TBG_nodes                          \
  --ntasks-per-node=1                         \
  --gres=gpu:0                                \
  --cpus-per-task=!TBG_coresPerPipeInstance   \
  openpmd-pipe                                \
    --infile "!TBG_streamdir"                 \
    --outfile "!TBG_dumpdir"                  \
    --inconfig @inconfig.json                 \
    --outconfig @outconfig.json               \
    > ../pipe.out 2> ../pipe.err              &

sleep 1

srun                                    \
  --exclusive                           \
  --ntasks !TBG_tasks                   \
  --nodes=!TBG_nodes                    \
  --ntasks-per-node=!TBG_devicesPerNode \
  --gres=gpu:!TBG_devicesPerNode        \
  --cpus-per-task=!TBG_coresPerGPU      \
  -K1                                   \
  !TBG_dstPath/input/bin/picongpu       \
    --mpiDirect                         \
    !TBG_author                         \
    !TBG_programParams                  \
    > ../pic.out 2> ../pic.err          &

wait
