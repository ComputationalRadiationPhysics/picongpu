# Copyright 2013-2021 Axel Huebl, Richard Pausch
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

if(OUTPUT2FILE == 1){
  set terminal postscript eps color "Helvetica" 20
  set grid
  set out 'OUTPUT_FILE.eps'
  }


set xlabel "E_n in MeV"
set ylabel "Number of PARTICLES"

set logscale y


plot "< cat \"FILENAME\" | awk '{if($1 == \"#step\" || $1 == \"TIMESTEP\") print}' | awk -f BINDIR/../share/awk/BinEnergyPlot.awk | egrep -v \"(>|<|step|count)\"" \
     u ($1/1000.0):2 t "PARTICLES at time step TIMESTEP" w l lw 2
