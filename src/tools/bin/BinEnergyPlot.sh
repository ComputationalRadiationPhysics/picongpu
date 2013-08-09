#!/bin/bash
# 
# Copyright 2013 Axel Huebl, René Widera
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

if test ! -f $1; then
 echo "$1 is not a file"
 echo "Usage:"
 echo "  $0 pathToSpectraFile.dat TIMESTEP ParticleNameForLegend"
 exit 1
fi

initCall="$0 $*"
bindir=`dirname $0`/

script=`cat $bindir/../share/gnuplot/BinEnergyPlot.gnuplot`

tmp1=`echo $1 | sed 's/\//\\\\\//g'`
tmp2=`echo $2 | bc | sed 's/\//\\\\\//g'`
tmp3=`echo $3 | sed 's/\//\\\\\//g'`
bindirEsc=`echo $bindir | sed 's/\//\\\\\//g'`

script=`echo "$script" | sed -e "s/FILENAME/"$tmp1"/g" | sed -e "s/TIMESTEP/"$tmp2"/g" | sed -e "s/BINDIR/"$bindirEsc"/g" | sed -e "s/PARTICLES/"$tmp3"/g"`

echo   "$script" | gnuplot -persist
