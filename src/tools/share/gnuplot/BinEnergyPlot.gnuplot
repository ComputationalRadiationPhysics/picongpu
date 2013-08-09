#!/usr/bin/gnuplot -persist
# 
# Copyright 2013 Axel Huebl
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

#set terminal postscript eps color "Helvetica" 20
#set grid
#set out 'BinEnergyElectrons.eps'

set xlabel "E_n in MeV"
set ylabel "Number of PARTICLES"

#MyYRange=3.0e-3

#set yrange [-MyYRange:MyYRange]
#set format y "%11.1e"

#set ytics MyYRange/5.0
set logscale y

# with lines
plot "< cat \"FILENAME\" | awk '{if($1 == \"#step\" || $1 == \"TIMESTEP\") print}' | awk -f BINDIR/../share/awk/BinEnergyPlot.awk | egrep -v \"(>|<|step|count)\"" \
     u ($1/1000.0):2 t "PARTICLES" w l lw 2
