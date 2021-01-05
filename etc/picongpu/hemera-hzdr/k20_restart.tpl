#!/usr/bin/env bash
# Copyright 2013-2021 Axel Huebl, Anton Helm, Rene Widera, Richard Pausch,
#                     Bifeng Lei, Marco Garten
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


## calculation are done by tbg ##
.TBG_queue="k20"

# settings that can be controlled by environment variables before submit
.TBG_mailSettings=${MY_MAILNOTIFY:-"n"}
.TBG_mailAddress=${MY_MAIL:-"someone@example.com"}
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}
.TBG_profile=${PIC_PROFILE:-"~/picongpu.profile"}

# number of available/hosted GPUs per node in the system
.TBG_numHostedGPUPerNode=4

# required GPUs per node for the current job
.TBG_gpusPerNode=`if [ $TBG_tasks -gt $TBG_numHostedGPUPerNode ] ; then echo $TBG_numHostedGPUPerNode; else echo $TBG_tasks; fi`

# host memory per gpu
.TBG_memPerGPU="$((62000 / $TBG_gpusPerNode))"
# host memory per node
.TBG_memPerNode="$((TBG_memPerGPU * TBG_gpusPerNode))"

# number of cores to block per GPU - we got 2 cpus per gpu
#   and we will be accounted 2 CPUs per GPU anyway
.TBG_coresPerGPU=2

# We only start 1 MPI task per GPU
.TBG_mpiTasksPerNode="$(( TBG_gpusPerNode * 1 ))"

# use ceil to caculate nodes
.TBG_nodes="$(( ( TBG_tasks + TBG_gpusPerNode -1 ) / TBG_gpusPerNode))"
## end calculations ##

# PIConGPU batch script for hemera's SLURM batch system

#SBATCH --partition=!TBG_queue
# necessary to set the account also to the queue name because otherwise access is not allowed at the moment
#SBATCH --account=!TBG_queue
#SBATCH --time=!TBG_wallTime
# Sets batch job's name
#SBATCH --job-name=!TBG_jobName
#SBATCH --nodes=!TBG_nodes
#SBATCH --ntasks=!TBG_tasks
#SBATCH --ntasks-per-node=!TBG_gpusPerNode
#SBATCH --mincpus=!TBG_mpiTasksPerNode
#SBATCH --cpus-per-task=!TBG_coresPerGPU
#SBATCH --mem=!TBG_memPerNode
#SBATCH --gres=gpu:!TBG_gpusPerNode
#SBATCH --mail-type=!TBG_mailSettings
#SBATCH --mail-user=!TBG_mailAddress
#SBATCH --chdir=!TBG_dstPath

# do not overwrite existing stderr and stdout files
#SBATCH --open-mode=append
#SBATCH -o stdout
#SBATCH -e stderr

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
sleep 1
cd simOutput

if [ ! -f output ]
then
    ln -s ../stdout output 2> /dev/null
fi

echo 'Running program...'
echo "----- automated restart routine -----"

#check whether last checkpoint is valid
file=""
# ADIOS restart files take precedence over HDF5 files
fileEnding="h5"
hasADIOS=$(ls ./checkpoints/checkpoint_*.bp 2>/dev/null | wc -w)
if [ $hasADIOS -gt 0 ]
then
    fileEnding="bp"
fi

for file in $(ls -t ./checkpoints/checkpoint_*.$fileEnding 2>/dev/null)
do
    echo -n "validate checkpoint $file: "
    $fileEnding"ls" $file &> /dev/null
    if [ $? -eq 0 ]
    then
        echo "OK"
        break
    else
        echo "FAILED $?"
        file=""
    fi
done

#this sed call extracts the final simulation step from the cfg (assuming a standard cfg)
finalStep=`echo !TBG_programParams | sed 's/.*-s[[:blank:]]\+\([0-9]\+[^\s]\).*/\1/'`
echo "final step      = " $finalStep
#this sed call extracts the -s and --checkpoint flags
programParams=`echo !TBG_programParams | sed 's/-s[[:blank:]]\+[0-9]\+[^\s]//g' | sed 's/--checkpoint\.period[[:blank:]]\+[0-9,:,\,]\+[^\s]//g'`
#extract restart period
restartPeriod=`echo !TBG_programParams | sed 's/.*--checkpoint\.period[[:blank:]]\+\([0-9,:,\,]\+[^\s]\).*/\1/'`
echo  "restart period = " $restartPeriod


# ******************************************* #
# need some magic, if the restart period is in new notation with the ':' and ','

if [ -z "$file" ]; then
    currentStep=0
else
    currentStep=`basename $file | sed 's/checkpoint_//g' | sed 's/.'$fileEnding'//g'`
fi

nextStep=$(nextstep_from_period.sh $restartPeriod $finalStep $currentStep)

if [ -z "$file" ]; then
    stepSetup="-s $nextStep --checkpoint.period $restartPeriod"
else
    stepSetup="-s $nextStep --checkpoint.period $restartPeriod --checkpoint.restart --checkpoint.restart.step $currentStep"
fi

# ******************************************* #

echo "--- end automated restart routine ---"

#wait that all nodes see ouput folder
sleep 1

# The OMPIO backend in OpenMPI up to 3.1.3 and 4.0.0 is broken, use the
# fallback ROMIO backend instead.
#   see bug https://github.com/open-mpi/ompi/issues/6285
export OMPI_MCA_io=^ompio

# test if cuda_memtest binary is available and we have the node exclusive
if [ -f !TBG_dstPath/input/bin/cuda_memtest ] && [ !TBG_numHostedGPUPerNode -eq !TBG_gpusPerNode ] ; then
  mpiexec !TBG_dstPath/input/bin/cuda_memtest.sh
else
  echo "Note: GPU memory test was skipped as no binary 'cuda_memtest' available or compute node is not exclusively allocated. This does not affect PIConGPU, starting it now" >&2
fi

if [ $? -eq 0 ] ; then
  mpiexec -tag-output --display-map !TBG_dstPath/input/bin/picongpu $stepSetup !TBG_author $programParams
fi

if [ $nextStep -lt $finalStep ]
then
    /usr/bin/sbatch "!TBG_dstPath/tbg/submit.start"
    if [ $? -ne 0 ] ; then
        echo "error during job submission"
    else
        echo "job submitted"
    fi
fi
