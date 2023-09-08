#!/usr/bin/env bash
# Copyright 2013-2023 Axel Huebl, Rene Widera, Felix Schmitt, Pawel Ordyna
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

## check if input files are newer than the picongpu binary
newerFiles=$(find $TBG_projectPath/include -type f \
  -newer $TBG_projectPath/bin/picongpu)
numNewerFiles=$(echo -e "$newerFiles" | wc -l)
if [ $numNewerFiles -gt 0 ] && [ -n "$newerFiles" ]
then
  (>&2 echo "WARNING: $numNewerFiles input file(s) in include/")
  (>&2 echo "         have been modified since the last compile!")
  (>&2 echo "         Did you forget to recompile?")
  (>&2 echo "         Run 'pic-build -f' to recompile with the modified files.")
  (>&2 echo "List of modified files:")
  (>&2 echo -e "$newerFiles")
fi

## copy memcheck programs
cd $TBG_dstPath
mkdir -p input
cp -ar $TBG_projectPath/bin input
cp -ar $TBG_projectPath/include input
cp -ar $TBG_projectPath/etc input
if [ -d "$TBG_projectPath/lib" ]
then
  cp -ar $TBG_projectPath/lib input
fi
if [ -f $TBG_projectPath/cmakeFlags ]
then
  cp -a $TBG_projectPath/cmakeFlags input
fi
if [ -f $TBG_projectPath/cmakeFlagsSetup ]
then
  cp -a $TBG_projectPath/cmakeFlagsSetup input
fi
cp -a $TBG_cfgPath/openib.conf tbg
cp -a $0 tbg
if [ -f $TBG_cfgPath/cpuNumaStarter.sh ]
then
  cp -a $TBG_cfgPath/cpuNumaStarter.sh tbg
fi
cp -a $TBG_cfgPath/handleSlurmSignals.sh tbg
