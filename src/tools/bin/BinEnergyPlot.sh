#!/usr/bin/env bash
#
# Copyright 2013-2021 Axel Huebl, Rene Widera, Richard Pausch
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

# set default output
OUTPUT2FILE=0
OUTPUT_FILE="BinEnergyElectrons"


function usage()
{
 programName=`basename $0`
 echo "Usage:"
 echo "  $programName [-h] [-l labelForLegend] [-o outputfile] pathToSpectraFile timeStep "
 echo
 echo "  -h                  - show usage of $programName"
 echo "  -l labelForLegend   - set name of label"
 echo "                        default \"$LABEL\""
 echo "  -o outputfile       - output to \"outputfile.eps\""
 echo "  pathToSpectraFile   - file containing Energy Histogram"
 echo "  timeStep            - time step to be shown"
}


# set up getopts

# Start processing of option flags at index 1 of all arguments.
OPTIND=1
# by setting OPTERR=1, error of the underlying getopt are reported
OPTERR=1

while getopts "hl:o:" FLAG "$@"
do
    # check for help flag
    if [ "$FLAG" = "h" ]
    then
        usage
        exit 0
    fi

    # check for label flag
    if [ "$FLAG" = "l" ]
    then
        LABEL=$OPTARG
    fi

    # check for output2file flag
    if [ "$FLAG" = "o" ]
    then
        OUTPUT2FILE=1
        if [ "$OPTARG" != "" ]
        then
            OUTPUT_FILE=$OPTARG
        fi
    fi

    # If the label after the -l or -o argument is missing, the getopt
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
# shift the argument list
shift `expr $OPTIND - 1`

# check whether the first argument is a file
if test ! -f $1; then
 echo "$1 is not a file"
 exit 1
fi


# directory of program
bindir=`dirname $0`/

# read out gnuplot script
script=`cat $bindir/../share/gnuplot/BinEnergyPlot.gnuplot`

# helper function to correctly escape paths in script replacement
escapeForLater()
{
    sed 's/\//\\\//g'
}

# prepare arguments for later sed replacements in gnuplot script
Esc1=`echo $1 | escapeForLater`
Esc2=`echo $2 | bc | escapeForLater`
LABELESC=`echo $LABEL | escapeForLater`
bindirEsc=`echo "$bindir" | escapeForLater`
OUTPUT_FILEEsc=`echo "$OUTPUT_FILE" | escapeForLater`

# adjust default gnuplot script
script=$( echo "$script" | sed -e "s/FILENAME/$Esc1/g" )
script=$( echo "$script" | sed -e "s/TIMESTEP/$Esc2/g" )
script=$( echo "$script" | sed -e "s/BINDIR/$bindirEsc/g" )
script=$( echo "$script" | sed -e "s/PARTICLES/$LABELESC/g" )
script=$( echo "$script" | sed -e "s/OUTPUT2FILE/$OUTPUT2FILE/g" )
script=$( echo "$script" | sed -e "s/OUTPUT_FILE/$OUTPUT_FILEEsc/g" )


# run gnuplot script
echo   "$script" | gnuplot -persist
