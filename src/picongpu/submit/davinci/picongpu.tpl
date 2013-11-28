#!/bin/bash
# Copyright 2013 Axel Huebl, Rene Widera, Richard Pausch
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
 

# PIConGPU batch script for DAVinCI (non-PRO) PBS batch system

#PBS -q TBG_queue
#PBS -l walltime=TBG_wallTime

# Sets batch job's name
#PBS -N TBG_shortJobName
#PBS -l nodes=TBG_nodes:ppn=TBG_GpusPerNode
# does not even work for interactive jobs
# :exclusive_process:reseterr
#PBS -W x=NACCESSPOLICY:SINGLEJOB
#PBS -V

# send me a mail on (b)egin, (e)nd, (a)bortion
##PBS -m TBG_mailSettings
##PBS -M TBG_mailAdress

#PBS -o TBG_outDir/stdout
#PBS -e TBG_outDir/stderr


echo 'Running program...'
echo TBG_jobName

cd TBG_outDir
echo -n "present working directory:"
pwd

# create hostfile with uniq histnames for makeParallelPictures
export LD_LIBRARY_PATH="TBG_outDir/build_libsplash:TBG_outDir/build_simlib:$LD_LIBRARY_PATH"

export MODULES_NO_OUTPUT=1

. /etc/profile.d/modules.sh
module load cuda/4.2.9
module load cmake
module load openmpi/1.4.4-gcc

unset MODULES_NO_OUTPUT

mkdir simOutput 2> /dev/null
cd simOutput

mpirun -n TBG_tasks --display-map -am tbg/openib.conf   ../picongpu/bin/picongpu TBG_programParams

