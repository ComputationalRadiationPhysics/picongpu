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
# Check for new commits and add them to the queue
# in queue.cnf

thisDir=$(cd `dirname $0` && pwd)"/"

. "$thisDir"config.sh
. "$thisDir"mails.sh

security_check

if [ -f "$thisDir"runGuard ] ; then
  echo "compileSuite already running..."
  exit 1
fi
touch "$thisDir"runGuard

# loop through each branch for new commits
#
cd $cnf_svndir

# loop branches
for b in "${cnf_branches[@]}"
do
    cd $b

    # loop commits
    finished=0
    while [ "$finished" -eq "0" ]
    do
        myRev=`svnversion`
        rev=$(( myRev + 1 ))
        "$thisDir"update_branch.sh $b $rev
        finished=$?

        # was this branch affected?
        if [ "$finished" -eq "0" ]; then
            logEntry=`svn log -v -r $rev`
            commitLine=`echo $logEntry | sed 's/-//g'`
            if [ -z "$commitLine" ] ; then
                finished=1
            fi
        fi

        # new version! :)
        if [ "$finished" -eq "0" ]; then
            # state of this test
            state=0 #0: suite errored, <0 compile failed, >0: ok

            # svn log infos
            logEntry=`svn log -v -r $rev`
            lastUser=`echo "$logEntry" | head -n 2 | tail -n 1 | awk -F"|" '{print $2}'`

            echo "Testing new version $b:$rev"

            cp $cnf_imgSrc $cnf_imgClone
            echo "Base System Image cloned..."

            echo "Starting Virtual Machine..."
            # -monitor stdio | -nographic
            /usr/bin/kvm -smp $cnf_numParallel -cpu kvm64 -enable-kvm \
                -m 2048 -nographic \
                -drive file="$cnf_imgClone",media=disk \
                -drive file="$cnf_extfile",media=disk \
                -boot once=c,menu=off -net none -name "Debian7_Cuda4_2"
            echo "Virtual Machine finished..."
            rm $cnf_imgClone

            echo "Error Analysis:"
            mntdir=$thisDir"tmp_mountResult"
            rm -rf $mntdir && mkdir -p $mntdir
            fuseext2 -o rw+ $cnf_extfile $mntdir

            # lastRun history
            if [ -f "$thisDir"lastRun.log ] ; then
                lastRun=`cat "$thisDir"lastRun.log`
            else
                touch "$thisDir"lastRun.log
                lastRun=0
            fi
            # is +/- integer?
            if ! [[ "$lastRun" =~ ^[\-0-9]+$ ]] ; then
                lastRun=0
            fi

            # analyse output
            returnCode=`cat $mntdir"/returnCode"`
            echo "Compile Suite return code: $returnCode"

            # is +/- integer?
            if ! [[ "$returnCode" =~ ^[\-0-9]+$ ]] ; then
                returnCode=1
                state=0 # suite errored (no return code)
                echo "NO return code! (Suite errored)"
            else
                if [ "$returnCode" -eq "0" ] ; then
                    echo "All right :)"
                    # last failed
                    if [ "$lastRun" -lt "0" ] ; then
                        state=1
                    # last was ok or suite errored
                    else
                        state=$(( lastRun + 1 ))
                    fi
                else
                    echo "Non-Zero return code!"
                    state=-1
                    # last failed
                    if [ "$lastRun" -lt "0" ] ; then
                        state=$(( lastRun - 1 ))
                    fi
                fi
            fi

            # update lastRun history
            echo $state > "$thisDir"lastRun.log

            # create conclusion and send mails
            conclusion "$state" "$lastUser" "$b" "$rev" "$logEntry" "$mntdir""/output"

            # unmount
            fusermount -u $mntdir
            while [ "$?" != "0" ]
            do
                sleep 5
                fusermount -u $mntdir
            done
            rm -rf $mntdir
        fi
    done

    cd $cnf_svndir
done

# del guard
rm "$thisDir"runGuard
