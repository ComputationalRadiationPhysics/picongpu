#!/usr/bin/env bash
# Copyright 2013-2023 Axel Huebl, Anton Helm, Richard Pausch, Rene Widera,
#                     Marco Garten, Pawel Ordyna
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


# PIConGPU batch script for hemera's SLURM batch system

#SBATCH --partition=!TBG_queue
# necessary to set the account also to the queue name because otherwise access is not allowed at the moment
#SBATCH --account=!TBG_account
#SBATCH --time=!TBG_wallTime
# Sets batch job's name
#SBATCH --job-name=!TBG_jobName
#SBATCH --nodes=!TBG_nodes
#SBATCH --ntasks=!TBG_tasks
#SBATCH --ntasks-per-node=!TBG_mpiTasksPerNode
#SBATCH --mincpus=!TBG_mpiTasksPerNode
#SBATCH --cpus-per-task=!TBG_coresPerGPU
#SBATCH --mem=!TBG_memPerNode
#SBATCH --gres=gpu:!TBG_gpusPerNode
#SBATCH --mail-type=!TBG_mailSettings
#SBATCH --mail-user=!TBG_mailAddress
#SBATCH --chdir=!TBG_dstPath
# notify the job 240 sec before the wall time ends
#SBATCH --signal=B:SIGALRM@240
#!TBG_keepOutputFileOpen

#SBATCH -o stdout
#SBATCH -e stderr

help()
{
  echo "PIConGPU submit script generated with tbg"
  echo ""
  echo "usage: $0 [--verify]"
  echo ""
  echo "--validate      - validate picongpu call instead of running the simulation"
  echo "--h | --help    - print this help message"
}

VALIDATE_MODE=false
for arg in "$@"; do
  case $arg in
  --validate)
    VALIDATE_MODE=true
    shift # Remove --skip-verification from `$@`
    ;;
  -h | --help)
    echo -e "$(help)"
    shift
    exit 0
    ;;
  *)
    echo "unrecognized argument"
    echo -e "$(help)"
    exit 1
    ;;
  esac
done

## calculations will be performed by tbg ##
.TBG_queue=${TBG_partition:-"k20"}
.TBG_account=`if [ $TBG_partition == "k20_low" ] ; then echo "low"; else echo "k20"; fi`
# configure if the output file should be appended or overwritten
.TBG_keepOutputFileOpen=`if [ $TBG_partition == "k20_low" ] ; then echo "SBATCH --open-mode=append"; fi`

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
.TBG_memPerGPU="$((62000 / $TBG_numHostedGPUPerNode))"
# host memory per node
.TBG_memPerNode="$((TBG_memPerGPU * TBG_gpusPerNode))"

# number of cores to block per GPU - we got 2 cpus per gpu
#   and we will be accounted 2 CPUs per GPU anyway
.TBG_coresPerGPU=2

# We only start 1 MPI task per GPU
.TBG_mpiTasksPerNode="$(( TBG_gpusPerNode * 1 ))"

# use ceil to caculate nodes
.TBG_nodes="$((( TBG_tasks + TBG_gpusPerNode - 1 ) / TBG_gpusPerNode))"

## end calculations ##

echo "Preparing environment..."

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

if [[ $VALIDATE_MODE == true ]]; then
   echo "Validating PIConGPU call..."
   !TBG_dstPath/input/bin/picongpu !TBG_author !TBG_programParams --validate
   if [ $? -ne 0 ] ; then
     exit 1;
   fi
else
    # test if cuda_memtest binary is available and we have the node exclusive
    if [ -f !TBG_dstPath/input/bin/cuda_memtest ] && [ !TBG_numHostedGPUPerNode -eq !TBG_gpusPerNode ] ; then
      # Run CUDA memtest to check GPU's health
      mpiexec -np !TBG_tasks !TBG_dstPath/input/bin/cuda_memtest.sh
    else
      echo "Note: GPU memory test was skipped as no binary 'cuda_memtest' available or compute node is not exclusively allocated. This does not affect PIConGPU, starting it now" >&2
    fi

    if [ $? -eq 0 ] ; then
      # Run PIConGPU
      echo "Running PIConGPU..."
      source !TBG_dstPath/tbg/handleSlurmSignals.sh mpiexec -np !TBG_tasks -tag-output --display-map !TBG_dstPath/input/bin/picongpu \
        !TBG_author !TBG_programParams
    fi
fi
