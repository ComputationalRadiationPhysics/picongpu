#!/bin/bash
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

if test ! -d "$1"; then
 echo "$1 is not a directory"
 echo "Usage:"
 echo "  $0 pathTolineSliceFieldFiles TIMESTEP [noPlot]"
 exit 1
fi

initCall="$0 $*"
bindir=`dirname $0`/

script=`cat $bindir/../share/gnuplot/LineSliceFields.gnuplot`

tmp1=`echo $1 | sed 's/\//\\\\\//g'`
tmp2=`echo $2 | sed 's/\//\\\\\//g'`

script=`echo "$script" | sed -e "s/PATH/"$tmp1"/g" | sed -e "s/TIMESTEP/"$tmp2"/g"`

rm -f $1/LSF_unsorted.dat $1/LSF.dat
export LC_NUMERIC=.

for i in $1lineSliceFields_*.txt
do
#  cat $i | grep "^$2 " | grep -v "^.* .* .* 0 0 0"
  cat $i | grep "^$2 " >> $1/LSF_unsorted.dat
#  cat $i | grep "^$2 " >> $1/LSF.dat
done

sort -gsk3 $1/LSF_unsorted.dat > $1/LSF.dat

############################
# FWHM Analyser
intenssum=`cat $1/LSF.dat | awk 'BEGIN{t=0.0} { t+=($4*$4) } END{printf("%.8e\n", t)}'`
intensmax=`cat $1/LSF.dat | awk 'BEGIN{t=0.0} { if(($4*$4)>t) t=($4*$4) } END{printf("%.8e\n", t)}'`
# with percentage (model perfect gaussian)
leftsigma=`cat $1/LSF.dat | awk -v max=$intenssum 'BEGIN{t=0.0; x=0.0} { t+=($4*$4); if(t/max>=0.15865 && x==0.0) x=$3 } END{printf("%.8e\n", x)}'`
rightsigma=`cat $1/LSF.dat | awk -v max=$intenssum 'BEGIN{t=0.0; x=0.0} { t+=($4*$4); if(t/max>=0.84135 && x==0.0) x=$3 } END{printf("%.8e\n", x)}'`

# in meters
twosigma=`echo "$rightsigma $leftsigma" | awk '{printf("%.8e\n", ($1-$2) )}'`
fwhm=`echo "$twosigma" | awk '{printf("%.8e\n", ($1*1.177410023) )}'`
# in seconds
sigmas=`echo "$twosigma" | awk '{printf("%.8e\n", ($1/2.0/299792458.0) )}'`
fwhms=`echo "$fwhm" | awk '{printf("%.8e\n", ($1/299792458.0) )}'`
err=`echo "$lambda" | awk '{printf("%.8e\n", ($1/299792458.0/2.0) )}'`

echo "intensmax: $intensmax [V^2/m^2]"
echo "sigma: $sigmas [s]"
echo "fwhm:  $fwhms [s]"
echo "+/- O($err [s] )   error: (integrate over lamda/2)"

#g++ LineSliceFieldsAnalyser.cpp -I../../libgpugrid/include -o LineSliceFieldsAnalyser.o
#./LineSliceFieldsAnalyser.o $1/LSF.dat
#echo "+/- O( 2% )     (some kind of low pass)"

python LineSliceFieldsAnalyserFFT.py $1/LSF.dat

############################
# GnuPlot
if test -z "$3"; then
  echo -e "$script" | gnuplot -persist
fi