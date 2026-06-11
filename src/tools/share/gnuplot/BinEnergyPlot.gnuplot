# SPDX-FileCopyrightText: Axel Huebl, Richard Pausch
#
# SPDX-License-Identifier: GPL-3.0-or-later

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
