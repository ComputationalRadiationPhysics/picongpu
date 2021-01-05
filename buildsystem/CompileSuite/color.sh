#!/usr/bin/env bash
#
# Copyright 2013-2021 Axel Huebl, Rene Widera
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

function echo_g {
    echo -e "\e[1;32m"$1"\e[0m"
}
function echo_b {
    echo -e "\e[0;34m"$1"\e[0m"
}
function echo_r {
    echo -e "\e[1;31m"$1"\e[0m"
}

compileSuite=`echo_b "[compileSuite] "`
compileError=`echo_r "[error] "`

function check {
  if test "$1" != "0"; then
    myError=1
    if [ -z "$2" ]; then
      echo $compileSuite`echo_r "$2"` >&2
    else
      echo $compileSuite`echo_r "   -> ERR!"` >&2
    fi
#  else
#    echo $compileSuite`echo_g "   -> OK"`
  fi
}

function thumbs_up {

    echo_g "\n
...........,_\n
........../.(|\n
..........\..\ \n
........___\..\,. ~~~~~~\n
.......(__)_)...\ \n
......(__)__)|...|\n
......(__)__)|.__|\n
.......(__)__)___/~~~~~~"
}

function thumbs_down {

    echo_r "\n
........_________\n
.......(__)__).__\~~~~~~\n
......(__)__)|...|\n
......(__)__)|...|\n
.......(__)_)..,/ \n
.........../../.. ~~~~~~\n
........../../\n
..........\_(|"
}
