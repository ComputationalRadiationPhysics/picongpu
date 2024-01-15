#!/usr/bin/env bash
# Copyright 2021-2023 Rene Widera
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

# This script is executing the expression given as parameters and forwards signals to the application.
# Signals will NOT only be forwarded, they will be mapped to use SLURM signals in a useful way.
#
# You need to source this script with your application as argument:
#   source handleSlurmSignals.sh foo.exe --foArg1="alice" --foArg2="bar"
#
# Signal mapping
#
# SIGTERM -> SIGUSR2
# SIGCONT -> SIGUSR1
# SIGUSR1 -> SIGUSR1
# SIGUSR2 -> SIGUSR2
# SIGALRM -> SIGUSR1 and SIGUSR2
#


fireSignal()
{
    for i in "$@"
    do
        kill -s $i $APP_PID
        echo "batch script: send signal $1 to $APP_PID" >&2
    done
}

trap "fireSignal SIGUSR2" SIGTERM
trap "fireSignal SIGUSR1" SIGCONT
trap "fireSignal SIGUSR1" SIGUSR1
trap "fireSignal SIGUSR2" SIGUSR2
trap "fireSignal SIGUSR1 SIGUSR2" SIGALRM

"$@" &
APP_PID=$!
echo "PID = ${APP_PID}"

while true
do
  wait $APP_PID
  kill -0 $APP_PID 2>/dev/null
  if [ $? -ne 0 ] ; then
    break;
  fi
done
