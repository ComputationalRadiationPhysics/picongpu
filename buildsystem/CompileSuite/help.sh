#!/usr/bin/env bash

# SPDX-FileCopyrightText: Axel Huebl
#
# SPDX-License-Identifier: GPL-3.0-or-later

#
#

help()
{
    echo "compile given examples"
    echo ""
    echo "usage: pic-compile [OPTION] src_dir dest_dir"
    echo ""
    echo "-l                   - interprete all folders in src_dir as examples and"
    echo "                       compile each of it"
    echo "-q                   - quiet run: don't ask the user and continue on errors"
    echo "                       but return a non-zero exit status"
    echo "-j <N>                 - spawn N tests in parallel (do not omit N)"
    echo "-c | --cmake         - overwrite options for cmake (e.g.: -c \"-DPIC_VERBOSE=1\")"
    echo "-h | --help          - show this help message"
    echo ""
    echo "Available environment vars:"
    echo "  "'$PIC_COMPILE_SUITE_CMAKE'" - example:"
    echo "  export PIC_COMPILE_SUITE_CMAKE=\"-DPIC_ENABLE_PNG=OFF -DPIC_ENABLE_HDF=OFF\""
    echo "Note: -c | --cmake will overwrite the environment variable."
    echo ""
    echo "Dependencies: dirname, basename"
}
