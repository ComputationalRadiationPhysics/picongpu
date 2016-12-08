#!/usr/bin/env bash
# Copyright 2013-2016 Axel Huebl, Anton Helm, Rene Widera, Richard Pausch
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
.TBG_queue="k80"

# settings that can be controlled by environment variables before submit
.TBG_mailSettings=${MY_MAILNOTIFY:-"n"}
.TBG_mailAddress=${MY_MAIL:-"someone@example.com"}
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}

# 8 gpus per node if we need more than 8 gpus else same count as TBG_tasks
.TBG_gpusPerNode=`if [ $TBG_tasks -gt 8 ] ; then echo 8; else echo $TBG_tasks; fi`

#number of cores per parallel node / default is 2 cores (4 HT threads) per gpu on k80 queue
.TBG_coresPerNode="$(( TBG_gpusPerNode * 2 ))"

# use ceil to caculate nodes
.TBG_nodes="$(( ( TBG_tasks + TBG_gpusPerNode -1 ) / TBG_gpusPerNode))"
## end calculations ##

# PIConGPU batch script for hypnos PBS batch system

#PBS -q !TBG_queue
#PBS -l walltime=!TBG_wallTime
# Sets batch job's name
#PBS -N !TBG_jobName
#PBS -l nodes=!TBG_nodes:ppn=!TBG_coresPerNode
# send me mails on job (b)egin, (e)nd, (a)bortion or (n)o mail
#PBS -m !TBG_mailSettings -M !TBG_mailAddress
#PBS -d !TBG_dstPath
#PBS -n

#PBS -o stdout
#PBS -e stderr

echo 'Running program...'

cd !TBG_dstPath

export MODULES_NO_OUTPUT=1
source ~/picongpu.profile
if [ $? -ne 0 ] ; then
  echo "Error: ~/picongpu.profile not found!"
  exit 1
fi
unset MODULES_NO_OUTPUT

#set user rights to u=rwx;g=r-x;o=---
umask 0027

mkdir simOutput 2> /dev/null
cd simOutput


sleep 1

echo "----- automated restart routine -----"

#check whether last checkpoint is valid
file=""
for file in `ls -t ./checkpoints/checkpoint_*.h5`
do
    echo -n "validate checkpoint $file: "
    h5ls $file &> /dev/null
    if [ $? -eq 0 ]
    then
        echo "OK"
        break
    else
        echo "FAILED"
        file=""
    fi
done

#this sed call extracts the final simulation step from the cfg (assuming a standard cfg)
finalStep=`echo !TBG_programParams | sed 's/.*-s[[:blank:]]\+\([0-9]\+[^\s]\).*/\1/'`
echo "final step      = " $finalStep
#this sed call extracts the -s and --checkpoint flags
programParams=`echo !TBG_programParams | sed 's/-s[[:blank:]]\+[0-9]\+[^\s]//g' | sed 's/--checkpoints[[:blank:]]\+[0-9]\+[^\s]//g'`
#extract restart period
restartPeriod=`echo !TBG_programParams | sed 's/.*--checkpoints[[:blank:]]\+\([0-9]\+[^\s]\).*/\1/'`
echo  "restart period = " $restartPeriod

if [ "" != "$file" ]
then
    cptimestep=`basename $file | sed 's/checkpoint_//g' | sed 's/.h5//g'`
    echo "start time      = " $cptimestep

    endTime="$(($cptimestep + $restartPeriod ))"
    echo "end time        = " $endTime

    stepSetup=$(echo -s $endTime "--restart --restart-step" $cptimestep "--checkpoints" $restartPeriod )
else
    echo "no checkpoint found"
    endTime=$restartPeriod
    stepSetup=$(echo " -s " $endTime "--checkpoints" $restartPeriod )
fi

echo "--- end automated restart routine ---"

#wait that all nodes see ouput folder
sleep 1

mpiexec --prefix $MPIHOME -tag-output --display-map -x LIBRARY_PATH -x LD_LIBRARY_PATH -am !TBG_dstPath/tbg/openib.conf --mca mpi_leave_pinned 0 -npernode !TBG_gpusPerNode -n !TBG_tasks !TBG_dstPath/picongpu/bin/cuda_memtest.sh

if [ $? -eq 0 ] ; then
  mpiexec --prefix $MPIHOME -x LIBRARY_PATH -x LD_LIBRARY_PATH -tag-output --display-map -am !TBG_dstPath/tbg/openib.conf --mca mpi_leave_pinned 0 -npernode !TBG_gpusPerNode -n !TBG_tasks !TBG_dstPath/picongpu/bin/picongpu $stepSetup !TBG_author $programParams | tee output
fi

mpiexec --prefix $MPIHOME -x LIBRARY_PATH -x LD_LIBRARY_PATH -npernode !TBG_gpusPerNode -n !TBG_tasks killall -9 picongpu 2>/dev/null || true

if [ $endTime -lt $finalStep ]
then
    ssh hypnos4 "/opt/torque/bin/qsub !TBG_dstPath/tbg/submit.start"
    if [ $? -ne 0 ] ; then
        echo "error during job submission"
    else
        echo "job submitted"
    fi
fi
