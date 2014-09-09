#!/bin/bash
#
# Copyright 2013-2014 Axel Huebl, Rene Widera, Richard Pausch
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

# info on usage of this program

# set default label
LABEL="electrons"

function usage()
{ 
 echo "Usage:"
 echo "  $0 [-h] [-l labelForLegend] pathToSpectraFile timeStep "
 echo 
 echo "  -h                  - show usage of $0"
 echo "  -l labelForLegend   - set name of label"
 echo "                        default \"$LABEL\""
 echo "  pathToSpectraFile   - file containg Energy Histogram"
 echo "  timeStep            - time step to be shown"
}


# set up optgets

# Start processing of option flags at index 1 of all arguments.
OPTIND=1
# by seting OPTERR=1, error of the underlying optget are  reported
OPTERR=1

while getopts "hl:" FLAG "$@"
do
    # check for help flag
    if [ "$FLAG" = "h" ]
    then
        usage
        exit 0
    fi

    #  check for label flag
    if [ "$FLAG" = "l" ]
    then
        LABEL=$OPTARG
    fi

    # If the label after the -l argument is missing, the getopt 
    # routine returns ":" 
    if [ "$VALUE" = ":" ]
    then
        echo "Flag -$OPTARG requires an argument."
        usage
        exit 2
    fi

    # If an unknown flag is used, getopts returns a "?"
    if [ "$VALUE" = "?" ]
    then
        echo "Unknown flag -$OPTARG"
        usage
        exit 2
    fi
done


# The first argument after the flags is at index $OPTIND
# shift the argumentlist 
shift `expr $OPTIND - 1`

# check whether the first  argument is a file
if test ! -f $1; then
 echo "$1 is not a file"
 exit 1
fi


# directory of program
bindir=`dirname $0`/

# read out gnuplot script
script=`cat $bindir/../share/gnuplot/BinEnergyPlot.gnuplot`

# helper function to  correctly escape paths in script replacement
escapeForLater()
{
    sed 's/\//\\\//g'
}

#  prepare arguments for later sed replacements in gnuplot script
tmp1=`echo $1 | escapeForLater`
tmp2=`echo $2 | bc | escapeForLater`
tmp3=`echo $LABEL | escapeForLater`
bindirEsc=`echo "$bindir" | escapeForLater`

# adjust default gnuplot script 
script=$( echo "$script" | sed -e "s/FILENAME/$tmp1/g" )
script=$( echo "$script" | sed -e "s/TIMESTEP/$tmp2/g" )
script=$( echo "$script" | sed -e "s/BINDIR/$bindirEsc/g" )
script=$( echo "$script" | sed -e "s/PARTICLES/$tmp3/g" )


# run gnuplot script
echo   "$script" | gnuplot -persist


