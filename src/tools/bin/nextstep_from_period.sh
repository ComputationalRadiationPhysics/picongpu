#!/usr/bin/env bash

# SPDX-FileCopyrightText: Axel Huebl, Ilja Goethel
#
# SPDX-License-Identifier: GPL-3.0-or-later

#
#

set -euf -o pipefail

helpText()
{
    echo "
    Usage: $0 period lastStep [currentStep]

    Script returns next timestep specified by the period syntax
    Takes three positional arguments:
     - a string representing period-syntax, like 3000,12000:15000:1000,1337:1337
     - and the final timestep (convertible to int)
     - the current timestep (convertible to int)

    and prints the next timestep to stdout"
}

# now we parse cmd params: first look for -h, then case depending on $#

for p in $@; do
    if [ "$p" = "-h" ] || [ "$p" = "--help" ]; then
        helpText
        exit 0
    fi
done

case $# in
    0 | 1)
        printf "\n *** Not enough arguments! *** \n"
        helpText
        exit 1
    ;;
    2)
        period="$1"
        lastStep="$2"
        currentStep=0
    ;;
    3)
        period="$1"
        lastStep="$2"
        currentStep="$3"
    ;;
    *)
        printf "\n *** Too many arguments! *** \n"
        helpText
        exit 1
    ;;
esac

splitSections()
{
    # split a string on occurences of ','
    IFS=','
    for s in $1; do
        echo $s
    done
}
expandPeriods()
{
    # split a string on occurences of ':' and transform it into a
    # sequence of checkpoint timesteps. Reads the global variable
    # $lastStep to determine the end of the sequence if not given
    # otherwise
    #   input syntax:
    #       period  or
    #       start:end[:period]
    #   examples: 5:20:5 -> 5 10 15 20
    #             3      -> 0 3 6 9   (with $lastStep == 10)
    #             3:5    -> 3 4 5
    IFS=':'
    input=($1)
    len=${#input[@]}
    case $len in
        1)
            seq 0 $1 $lastStep
        ;;
        2)
            seq ${input[0]} 1 ${input[1]}
        ;;
        3)
            seq ${input[0]} ${input[2]} ${input[1]}
        ;;
        *)
            exit 1
        ;;
    esac
}

sections=$(splitSections "$period")
steps=()
for s in $sections; do
    newSteps=($(expandPeriods $s))
    steps+=(${newSteps[@]})
done

steps=($(printf "%s\n" "${steps[@]}" | sort -u -g))

for step in "${steps[@]}"; do
    if [ $step -gt $currentStep ]; then
        echo $step
        break
    fi
done
