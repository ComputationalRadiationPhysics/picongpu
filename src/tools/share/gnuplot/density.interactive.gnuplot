# 
# Copyright 2013 Axel Huebl, Rene Widera
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

reset
infile=system("echo $INFILE")

set palette defined ( 0 "white" ,  0.1 '#000fff',0.2 '#0090ff',0.3 '#0fffee',0.4 '#90ff70',0.5 '#ffee00',0.6 '#ff7000',0.7 '#ee0000', 1 '#7f0000')
#set palette defined ( 0 '#000090', 1 '#000fff', 2 '#0090ff', 3 '#0fffee', 4 '#90ff70', 5 '#ffee00', 6 '#ff7000',  7 '#ee0000', 8 '#7f0000')

x_axis=system("echo $X_AXIS")
y_axis=system("echo $Y_AXIS")
colormax=system("echo $COLORMAX")

set cblabel "Number of particles per cubic meter" offset 0.5,0,0
set ylabel y_axis 
set xlabel x_axis 

set terminal dumb
plot infile nonuniform matrix using 1:2:3 with image title ""

set terminal wxt size 800,600 enhanced font 'Verdana,14' persist

set border linewidth 2
set pointsize 5


set format x "%.1te%S"
set format y "%.1te%S"
set format cb "%.1te%S"

if (colormax==0)  colormax=GPVAL_DATA_CB_MAX ; else colormax=colormax

set xrange [GPVAL_DATA_X_MIN:GPVAL_DATA_X_MAX]
set yrange [GPVAL_DATA_Y_MIN:GPVAL_DATA_Y_MAX]
set cbrange [0:colormax]


replot
