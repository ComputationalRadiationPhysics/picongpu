#!/bin/bash
# Copyright 2013 Rene Widera
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
 
# PIConGPU batch script for hypnos PBS batch system

#MSUB -q TBG_queue
#MSUB -T TBG_wallTime
# Sets batch job's name
#MSUB  -r TBG_jobName
#MSUB  -N TBG_nodes
#MSUB  -n TBG_tasks
#MSUB  -c TBG_cores
##MSUB  -@ TBG_mailAdress
#MSUB  -E "-cwd TBG_outDir"
#MSUB -p TBG_userGroup

#MSUB -o stdout
#MSUB -e stderr


echo 'Running program...'

cd TBG_outDir

# create hostfile with uniq histnames for makeParallelPictures
cat $LSB_DJOB_HOSTFILE | sort | uniq > host

export LD_LIBRARY_PATH="TBG_outDir/build_libsplash:TBG_outDir/build_simlib:$LD_LIBRARY_PATH"


##load modules

###please add here your modules
mkdir simOutput 2> /dev/null
cd simOutput

mpirun  --display-map -am openib.conf  -n TBG_tasks ../picongpu/bin/picongpu TBG_programParams
