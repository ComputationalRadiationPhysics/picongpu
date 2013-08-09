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

# global functions ############################################################
#

# create conclusion
# $1: state
#   0 suite errored
#  <0 compile failed
#  >0 compile ok
#
# $2: lastUser
# $3: branch
# $4: rev
# $5: logEntry
# $6: path to compile output file
function conclusion {
    state="$1"
    echo "State: $state"
    lastUser="$2"
    #echo "lastUser: $lastUser"
    branch="$3"
    #echo "branch: $branch"
    rev="$4"
    #echo "rev: $rev"
    logEntry="$5"
    #echo "logEntry: $logEntry"
    compileOutputFile="$6"
    #echo "compile (tail): "
    #tail -n 10 $compileOutputFile

    subject=""
    text=""
    # Positive
    if [ "$state" -gt "0" ] ; then
        # "award" mail?
        award=$(( state % cnf_congrats ))
        if [ "$award" -eq "0" ] ; then
            subject="[Award] $lastUser @ $branch : $rev"
            text="You won a *team award*! $cnf_congrats in a row! *congrats*! :)"
        fi

        # problem "fixed" mail?
        if [ "$state" -eq "1" ] ; then
            subject="[Fixed] $lastUser @ $branch : $rev"
            text="$lastUser *fixed* PIConGPU! We love you!"
        fi
    fi

    # Test Failed - inform users
    if [ "$state" -lt "0" ] ; then
        # first fail
        if [ "$state" -eq "-1" ] ; then
            subject="[Failed] $lastUser @ $branch : $rev"
            text="_Errors_ occured! Dare you *$lastUser*! Pls fix them ... Allez garcon!"
        # still failing
        else
            subject="[Still Failing] $lastUser @ $branch : $rev"
            text="_Errors_ occured! Compile *still* failing ($lastUser did _not_ fix all errors...)"
        fi
        # parse errors
        text="$text

*Failing* Tests:"
        fTests=`grep -iR "\[compileSuite\] \[error\]" "$compileOutputFile" | awk -F':' '{print $2}' | awk -F'=' '{print $2}'`

        # for each error loop and show first 35 lines ...
        for fT in $fTests
        do
            text="$text

$fT
"
            lnStart=`grep -n "$fT" "$compileOutputFile" | head -n1 | awk -F':' '{print $1}'`
            lnEnd=$(( lnStart + cnf_mail_preview ))
            echo "$fT : $lnStart - $lnEnd"
            text="$text"`awk -v a=$lnStart -v b=$lnEnd 'NR>=a && NR<=b {print "> ",$0}' $compileOutputFile`
        done
    fi

    # Suite Errored - internal error
    if [ "$state" -eq "0" ] ; then
        subject="[Errored] $lastUser @ $branch : $rev"
        text="Compile Suite: internal error"
    fi

    text="$text

*LogEntry*:
$logEntry"

    # send mail
    echo "Mail Subject: $subject"
    #echo "Mail Text: $text"
    #echo "Mail attachement: $compileOutputFile"

    if [ ! -z "$subject" ] ; then
        send_mail "$subject" "$text" "$compileOutputFile"
    fi
}

# send mail via mailx
#   debian: apt-get install heirloom-mailx
# $1: subject
# $2: body
# $3: attachement
#
function send_mail()
{
    subject="[CompileSuite] $1"
    export smtp="$cnf_smtp"
    #export smtp-auth-user="$smtp-auth-user"
    #export smtp-auth-password="$cnf_smtp-auth-password"
    #sendcharsets=...
    export from="$cnf_from"
    export rctp_to="$cnf_rctp_to"

    if [ ! -z "$3" ] && [ -f "$3" ] ; then
      echo "$2" | mailx -a "$3" -s "$subject" "$rctp_to"
    else
      echo "$2" | mailx -s "$subject" "$rctp_to"
    fi

    echo "Mail send!"
}
