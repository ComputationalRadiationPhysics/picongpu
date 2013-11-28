#!/bin/bash
# Copyright 2013 Axel Huebl, Rene Widera, Richard Pausch, Wen Fu
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
TBG_gpuType="m2050"
TBG_queue="largemem"
TBG_mailSettings="bea"
TBG_mailAdress="someone@example.com"
#number of cores per parallel node / default is 2 cores per gpu on k20 queue
    
# 2 gpus per node if we need more than 2 gpus else same count as TBG_tasks   
TBG_gpusPerNode=`if [ $TBG_tasks -gt 2 ] ; then echo 2; else echo $TBG_tasks; fi`

# use one core per gpu    
TBG_coresPerNode=$TBG_gpusPerNode
    
# use ceil to caculate nodes
TBG_nodes="$(( ( TBG_tasks + TBG_gpusPerNode -1 ) / TBG_gpusPerNode))"
## end calculations ##

# PIConGPU batch script for judge moab batch system

#MSUB -q !TBG_queue
#MSUB -l walltime=!TBG_wallTime
#Sets batch job's name
#MSUB -N !TBG_jobName
#MSUB -l nodes=!TBG_nodes:ppn=!TBG_coresPerNode:gpus=!TBG_gpusPerNode:!TBG_gpuType
#MSUB -l mem=20gb
#MSUB -l pmem=10gb
##MSUB -l naccesspolicy=singlejob
#send me a mail on (b)egin, (e)nd, (a)bortion
##MSUB -m !TBG_mailSettings -M !TBG_mailAdress
#MSUB -d !TBG_dstPath

#MSUB -o stdout
#MSUB -e stderr    
    
    
echo 'Running program...'
echo !TBG_jobName

cd !TBG_dstPath
echo -n "present working directory:"
pwd


export MODULES_NO_OUTPUT=1

export CUDA_ROOT=$CUDAROOT
export CUDA_LIB=$CUDAROOT/lib64
export SPLASH_ROOT=~/src/splash/src
export PNGWRITER_ROOT=~/lib/pngwriter
export HDF5_ROOT=~/lib/hdf5
export HDF5_INC=$HDF5_ROOT/include
export HDF5_LIB=$HDF5_ROOT/lib
export LD_LIBRARY_PATH=$BOOST_LIB_EXT:$LD_LIBRARY_PATH:$PNGWRITER_ROOT/lib:$CUDA_ROOT/lib64/:$CUDA_ROOT/lib:$HDF5_ROOT/lib:$MPI_ROOT/lib:$BOOST_LIB


mkdir simOutput 2> /dev/null
cd simOutput

mpiexec  -np !TBG_tasks  !TBG_dstPath/picongpu/bin/cuda_memtest.sh

if [ $? -eq 0 ] ; then
   mpiexec  -np !TBG_tasks  !TBG_dstPath/picongpu/bin/picongpu !TBG_programParams
fi 
