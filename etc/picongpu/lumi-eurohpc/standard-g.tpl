#!/usr/bin/env bash
# Copyright 2013-2023 Axel Huebl, Richard Pausch, Rene Widera, Sergei Bastrakov, Klaus Steinger
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


# PIConGPU batch script for frontier's SLURM batch system

#SBATCH --account=!TBG_nameProject
#SBATCH --partition=!TBG_queue
#SBATCH --time=!TBG_wallTime
# Sets batch job's name
#SBATCH --job-name=!TBG_jobName
#SBATCH --nodes=!TBG_nodes_adjusted
#SBATCH --ntasks=!TBG_tasks_adjusted
#SBATCH --gpus-per-node=!TBG_devicesPerNode
#SBATCH --mem-per-gpu=!TBG_memPerDevice
#SBATCH --gres=gpu:!TBG_devicesPerNode

#SBATCH --mail-type=!TBG_mailSettings
#SBATCH --mail-user=!TBG_mailAddress
#SBATCH --chdir=!TBG_dstPath

#SBATCH --open-mode=append
#SBATCH -o stdout
#SBATCH -e stderr

## calculations will be performed by tbg ##
.TBG_queue="standard-g"

# settings that can be controlled by environment variables before submit
.TBG_mailSettings=${MY_MAILNOTIFY:-"NONE"}
.TBG_mailAddress=${MY_MAIL:-"someone@example.com"}
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}
.TBG_nameProject=${PROJID:-""}
.TBG_profile=${PIC_PROFILE:-"~/picongpu.profile"}

# number of available/hosted devices per node in the system
.TBG_numHostedDevicesPerNode=8

# host memory per device
# 128 GiB per NUMA = 2 GCD - but only 60 GiB per GCD available
.TBG_memPerDevice=60000

# number of CPU cores to block per GPU
# we have 8 CPU cores per GPU (64cores/8gpus ~ 8cores)
# but one core (= core 0) is reserved for system processes
# called low-noise mode - see LUMI-G documentation
.TBG_coresPerGPU=7

# required GPUs per node for the current job
.TBG_devicesPerNode=$(if [ $TBG_tasks -gt $TBG_numHostedDevicesPerNode ] ; then echo $TBG_numHostedDevicesPerNode; else echo $TBG_tasks; fi)

# We only start 1 MPI task per device
.TBG_mpiTasksPerNode="$(( TBG_devicesPerNode * 1 ))"

# use ceil to caculate nodes
.TBG_nodes="$((( TBG_tasks + TBG_devicesPerNode - 1 ) / TBG_devicesPerNode))"

# oversubscribe the node allocation by N per thousand
# The default can be overwritten by setting the environment variable PIC_NODE_OVERSUBSCRIPTION_PT
.TBG_node_oversubscription_pt=${PIC_NODE_OVERSUBSCRIPTION_PT:-2}

# adjust number of nodes for fault tolerance adjustments
.TBG_nodes_adjusted=$((!TBG_nodes * (1000 + !TBG_node_oversubscription_pt) / 1000))
.TBG_tasks_adjusted=$((!TBG_nodes_adjusted * !TBG_numHostedDevicesPerNode))

## end calculations ##

echo 'Start job with !TBG_nodes_adjusted nodes. Required are !TBG_nodes nodes.'

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

# number of broken nodes
n_broken_nodes=0

# return code of cuda_memcheck
node_check_err=1

if [ -f !TBG_dstPath/input/bin/cuda_memtest ] && [ !TBG_numHostedDevicesPerNode -eq !TBG_mpiTasksPerNode ] ; then
    run_cuda_memtest=1
else
    run_cuda_memtest=0
fi


cat << EOF > select_gpu
#!/bin/bash

export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
exec "\$@"
EOF

chmod +x ./select_gpu

CPU_BIND="mask_cpu:7e000000000000,7e00000000000000"
CPU_BIND="${CPU_BIND},7e0000,7e000000"
CPU_BIND="${CPU_BIND},7e,7e00"
CPU_BIND="${CPU_BIND},7e00000000,7e0000000000"

export OMP_NUM_THREADS=6
export MPICH_GPU_SUPPORT_ENABLED=1


# test if cuda_memtest binary is available and we have the node exclusive
if [ $run_cuda_memtest -eq 1 ] ; then
    touch bad_nodes.txt
    n_tasks=$((!TBG_nodes_adjusted * !TBG_numHostedDevicesPerNode))
    for((i=0; ($n_tasks >= !TBG_tasks) && ($node_check_err != 0); ++i)) ; do
        n_tasks_last_check=$n_tasks
        mkdir "cuda_memtest_$i"
        cd "cuda_memtest_$i"
        # Run cuda_memtest (HIP version) to check GPU's health
        echo "GPU memtest started with $n_tasks tasks. Required are !TBG_tasks tasks."
        test $n_broken_nodes -ne 0 && exclude_nodes="-x../bad_nodes.txt"
        # do not bind to any GPU, else we can not use the local MPI rank to select a GPU
        # - test always all except the broken nodes
        # - catch error to avoid that the batch script stops processing in case an error happened
        node_check_err=$(srun -n $n_tasks --nodes=$((n_tasks / !TBG_numHostedDevicesPerNode)) $exclude_nodes -K1 --gpu-bind=none !TBG_dstPath/input/bin/cuda_memtest.sh && echo 0 || echo 1)
        cd ..
        ls -1 "cuda_memtest_$i" | sed -n -e 's/cuda_memtest_\([^_]*\)_.*/\1/p' | sort -u >> ./bad_nodes.txt
        n_broken_nodes=$(cat ./bad_nodes.txt | sort -u | wc -l)
        n_tasks=$(((!TBG_nodes_adjusted - n_broken_nodes) * !TBG_numHostedDevicesPerNode))
        # if cuda_memtest not passed and we have no broken nodes something else went wrong
        if [ $node_check_err -ne 0 ] ; then
            if [ $n_tasks_last_check -eq $n_tasks ] ; then
                echo "cuda_memtest: Number of broken nodes has not increased but for unknown reasons cuda_memtest reported errors." >&2
                break
            fi
            if [ $n_broken_nodes -eq 0 ] ; then
                echo "cuda_memtest: unknown error" >&2
                break
            else
                echo "cuda_memtest: "$n_broken_nodes" broken node(s) detected!. The test will be repeated with healthy nodes only." >&2
            fi
        fi
    done
    echo "GPU memtest with $n_tasks tasks finished with error code $node_check_err."
else
    echo "Note: GPU memory test was skipped as no binary 'cuda_memtest' available or compute node is not exclusively allocated. This does not affect PIConGPU, starting it now" >&2
fi

if [ $node_check_err -eq 0 ] || [ $run_cuda_memtest -eq 0 ] ; then
    # Run PIConGPU
    echo "Start PIConGPU."
    test $n_broken_nodes -ne 0 && exclude_nodes="-x./bad_nodes.txt"

    srun --cpu-bind=${CPU_BIND}  -n !TBG_tasks --nodes=!TBG_nodes $exclude_nodes -K1 ./select_gpu !TBG_dstPath/input/bin/picongpu --mpiDirect !TBG_author !TBG_programParams
else
    echo "Job stopped because of previous issues."
    echo "Job stopped because of previous issues." >&2
fi

rm -rf ./select_gpu
