#!/usr/bin/env bash
# Copyright 2013-2021 Axel Huebl, Richard Pausch, Alexander Debus, Klaus Steiniger
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
# This tpl for automated restarts is older than the actual
# V100.tpl.
# It uses a machine file for parallel job execution,
# which is not necessary anymore.
# (See comment below, MPI has been fixed)
# However, it still works and therefore is left unchanged.
# Klaus, June 2019

#SBATCH --partition=!TBG_queue
#SBATCH --time=!TBG_wallTime
# Sets batch job's name
#SBATCH --job-name=!TBG_jobName
#SBATCH --nodes=!TBG_nodes
#SBATCH --ntasks=!TBG_tasks
#SBATCH --ntasks-per-node=!TBG_gpusPerNode
#SBATCH --mincpus=!TBG_mpiTasksPerNode
#SBATCH --cpus-per-task=!TBG_coresPerGPU
# Maximum memory setting the SLURM queue "ml" accepts.
#SBATCH --mem=0
#SBATCH --gres=gpu:!TBG_gpusPerNode
#SBATCH --exclusive

# disable hyperthreading (default on taurus)
#SBATCH --hint=nomultithread

# send me mails on BEGIN, END, FAIL, REQUEUE, ALL,
# TIME_LIMIT, TIME_LIMIT_90, TIME_LIMIT_80 and/or TIME_LIMIT_50
#SBATCH --mail-type=!TBG_mailSettings
#SBATCH --mail-user=!TBG_mailAddress
#SBATCH --chdir=!TBG_dstPath

#SBATCH -o stdout
#SBATCH -e stderr


## calculations will be performed by tbg ##
.TBG_queue="ml"

# settings that can be controlled by environment variables before submit
.TBG_mailSettings=${MY_MAILNOTIFY:-"ALL"}
.TBG_mailAddress=${MY_MAIL:-"someone@example.com"}
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}
.TBG_profile=${PIC_PROFILE:-"~/picongpu.profile"}

# 6 gpus per node
.TBG_gpusPerNode=`if [ $TBG_tasks -gt 6 ] ; then echo 6; else echo $TBG_tasks; fi`

# number of CPU cores to block per GPU
# we got 7 CPU cores per GPU (44cores/6gpus ~ 7cores)
.TBG_coresPerGPU=7

# We only start 1 MPI task per GPU
.TBG_mpiTasksPerNode="$(( TBG_gpusPerNode * 1 ))"

# use ceil to calculate nodes
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

# set user rights to u=rwx;g=r-x;o=---
umask 0027

# Due to missing SLURM integration of the current MPI libraries
# we have to create a suitable machinefile.
rm -f machinefile.txt
for i in `seq !TBG_gpusPerNode`
do
    scontrol show hostnames $SLURM_JOB_NODELIST >> machinefile.txt
done

mkdir simOutput 2> /dev/null
cd simOutput

# we are not sure if the current bullxmpi/1.2.4.3 catches pinned memory correctly
#   support ticket [Ticket:2014052241001186] srun: mpi mca flags
#   see bug https://github.com/ComputationalRadiationPhysics/picongpu/pull/438
export OMPI_MCA_mpi_leave_pinned=0
# Use ROMIO for IO
# according to ComputationalRadiationPhysics/picongpu#2857
export OMPI_MCA_io=^ompio

sleep 1

echo "----- automated restart routine -----" | tee -a output

#check whether last checkpoint is valid
file=""
# ADIOS restart files take precedence over HDF5 files
fileEnding="h5"
hasADIOS=$(ls ./checkpoints/checkpoint_*.bp 2>/dev/null | wc -w)
if [ $hasADIOS -gt 0 ]
then
    fileEnding="bp"
fi

for file in `ls -t ./checkpoints/checkpoint_*.$fileEnding`
do
    echo -n "validate checkpoint $file: " | tee -a output
    $fileEnding"ls" $file &> /dev/null
    if [ $? -eq 0 ]
    then
        echo "OK" | tee -a output
        break
    else
        echo "FAILED" | tee -a output
        file=""
    fi
done

#this sed call extracts the final simulation step from the cfg (assuming a standard cfg)
finalStep=`echo !TBG_programParams | sed 's/.*-s[[:blank:]]\+\([0-9]\+[^\s]\).*/\1/'`
echo "final step      = " $finalStep | tee -a output
#this sed call extracts the -s and --checkpoint flags
programParams=`echo !TBG_programParams | sed 's/-s[[:blank:]]\+[0-9]\+[^\s]//g' | sed 's/--checkpoint\.period[[:blank:]]\+[0-9,:,\,]\+[^\s]//g'`
#extract restart period
restartPeriod=`echo !TBG_programParams | sed 's/.*--checkpoint\.period[[:blank:]]\+\([0-9,:,\,]\+[^\s]\).*/\1/'`
echo  "restart period = " $restartPeriod | tee -a output


# ******************************************* #
# need some magic, if the restart period is in new notation with the ':' and ','

currentStep=`basename $file | sed 's/checkpoint_//g' | sed 's/.'$fileEnding'//g'`
nextStep=$(nextstep_from_period.sh $restartPeriod $finalStep $currentStep)

if [ -z "$file" ]; then
    stepSetup="-s $nextStep --checkpoint.period $restartPeriod"
else
    stepSetup="-s $nextStep --checkpoint.period $restartPeriod --checkpoint.restart --checkpoint.restart.step $currentStep"
fi

# ******************************************* #

echo "--- end automated restart routine ---" | tee -a output

#wait that all nodes see output folder
sleep 1

# test if cuda_memtest binary is available
if [ -f !TBG_dstPath/input/bin/cuda_memtest ] ; then
  # Run CUDA memtest to check GPU's health
  mpiexec -hostfile ../machinefile.txt !TBG_dstPath/input/bin/cuda_memtest.sh
else
  echo "Note: GPU memory test was skipped as no binary 'cuda_memtest' available. This does not affect PIConGPU, starting it now" >&2
fi

if [ $? -eq 0 ] ; then
  # Run PIConGPU
  mpiexec -hostfile ../machinefile.txt !TBG_dstPath/input/bin/picongpu $stepSetup !TBG_author !TBG_programParams | tee output
fi

mpiexec -hostfile ../machinefile.txt /usr/bin/env bash -c "killall -9 picongpu 2>/dev/null || true"

if [ $nextStep -lt $finalStep ]
then
    ssh tauruslogin6 "/usr/bin/sbatch !TBG_dstPath/tbg/submit.start"
    if [ $? -ne 0 ] ; then
        echo "error during job submission" | tee -a output
    else
        echo "job submitted" | tee -a output
    fi
fi

