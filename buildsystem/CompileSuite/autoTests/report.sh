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

# global functions ############################################################
#

# create conclusion
# $1: state
#   0 suite errored
#  <0 compile failed
#  >0 compile ok
#
# $2: lastUser
# $3: lastUserMail
# $4: sha
# $5: eventid
# $6: logEntry
# $7: path to compile output file
function conclusion {
    state="$1"
    echo "State: $state"
    lastUser="$2"
    echo "lastUser: $lastUser"
    lastUserMail="$3"
    #echo "email: $lastUserMail"
    sha="$4"
    echo "sha: $sha"
    eventid="$5"
    echo "eventid: $eventid"
    logEntry="$6"
    #echo "logEntry: $logEntry"
    compileOutputFile="$7"
    #echo "compile (tail): "
    #tail -n 10 $compileOutputFile

    subject=""
    text=""
    stateName=""
    # Positive
    if [ "$state" -gt "0" ] ; then
        stateName="success"
        subject="[Build OK] $lastUser @ $sha"
        text="Tested commit from $lastUser was clean! Well done yo!"
    fi

    # Test Failed - inform users
    if [ "$state" -lt "0" ] ; then
        stateName="failure"
        subject="[Failed] $lastUser @ $sha"
        text="_Errors_ occured! Dare you *$lastUser*! Pls fix them ... Allez garcon!"
        # parse errors
        text="$text

*Failing* Tests:"
        fTests=`grep -iR "\[compileSuite\] \[error\]" "$compileOutputFile" | awk -F':' '{print $2}' | awk -F'=' '{print $2}'`

        # for each error loop and show first N lines ...
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
        stateName="error"
        subject="[Errored] $lastUser @ $sha"
        text="Compile Suite: internal error"
    fi

    text="$text

*LogEntry*:
$logEntry"

    echo "Summary: $subject"

    # report to scheduler
    #
    # escape special characters
    textJSON=`echo -n "$text" | python -c 'import json,sys; print json.dumps(sys.stdin.read())'`
    postParams='{"action":"report","eventid":'$eventid',"result":"'$stateName'","output":'$textJSON'}'
    echo -n "$postParams" > "$thisDir"lastPostParams.log
    curl -s -X POST --data-urlencode payload@"$thisDir"lastPostParams.log $cnf_scheduler
    if [ $? -ne 0 ]; then
        echo "Error contacting scheduler at $cnf_scheduler"
        exit 1
    fi
}
