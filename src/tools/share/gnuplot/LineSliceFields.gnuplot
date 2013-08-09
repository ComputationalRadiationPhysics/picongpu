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

set xlabel "y   in meters"
set ylabel "E_x^2   in V^2 * m^-2"

#MyYRange=3.0e-3

set format x "%11.1e"
#set yrange [-MyYRange:MyYRange]
#set xrange [0.0:1.0e-5]
set format y "%11.1e"

#set ytics MyYRange/5.0

# to do: kind of "sort by y-position" for lineplots
plot 'PATHLSF.dat' using ($3):(($4)*($4)) title "E_x^2" with p