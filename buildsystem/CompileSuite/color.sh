#!/usr/bin/env bash

# SPDX-FileCopyrightText: Axel Huebl, Rene Widera
#
# SPDX-License-Identifier: GPL-3.0-or-later

#
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
