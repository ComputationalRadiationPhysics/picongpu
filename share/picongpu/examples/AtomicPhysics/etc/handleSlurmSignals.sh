#!/usr/bin/env bash

# SPDX-FileCopyrightText: Rene Widera
#
# SPDX-License-Identifier: GPL-3.0-or-later

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
