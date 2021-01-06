#!/usr/bin/env bash
#
# Copyright 2013-2021 Axel Huebl
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
