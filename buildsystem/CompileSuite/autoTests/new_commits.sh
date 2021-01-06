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
# Check for new commits and add them to the queue
# in queue.cnf

thisDir=$(cd `dirname $0` && pwd)"/"

. "$thisDir"config.sh
. "$thisDir"report.sh

security_check

if [ -f "$thisDir"runGuard ] ; then
  echo "compileSuite already running..."
  exit 1
fi
touch "$thisDir"runGuard

# loop branches
#for b in "${cnf_branches[@]}"
#do
#    cd $b

    # loop work
    finished=0
    #while [ "$finished" -eq "0" ]
    #do
        cd $cnf_gitdir

        "$thisDir"get_work.sh
        finished=$?

        # merge conflict occuted
        if [ "$finished" -eq "2" ]; then
            # clean message as failure
            echo "Merge conflict detected. Aborting..."

            state=-1
            # git log infos of TO-BE-MERGED branch (that failed)
            sha=`cat .git/MERGE_HEAD`
            logEntry="*Merge Conflict Detected*"`echo && git log $sha -1`
            lastUser=`git log $sha -1 --format=%an`
            lastUserMail=`git log $sha -1 --format=%ae`
            if [ -z "$lastUserMail" ] ; then
                lastUserMail="example@example.com"
            fi
            eventid=`cat "$thisDir"runGuard`

            # create conclusion, update status (and send mails)
            conclusion "$state" "$lastUser" "$lastUserMail" "$sha" "$eventid" "$logEntry" "$thisDir""runGuard"
        fi

        # new version! :)
        if [ "$finished" -eq "0" ]; then
            # state of this test
            state=0 #0: suite errored, <0 compile failed, >0: ok

            # git log infos
            logEntry=`git log -1`
            lastUser=`git log -1 --format=%an`
            lastUserMail=`git log -1 --format=%ae`
            if [ -z "$lastUserMail" ] ; then
                lastUserMail="example@example.com"
            fi
            sha=`git log -1 --format=%H`
            eventid=`cat "$thisDir"runGuard`

            echo "Testing new commit"

            # create clean build dir
            rm -rf $cnf_builddir
            mkdir -p $cnf_builddir
            cd $cnf_builddir

            # modify compile environment (forwarded to CMake)
            #export PIC_COMPILE_SUITE_CMAKE="-DPIC_ENABLE_PNG=OFF -DALPAKA_CUDA_ARCH=35"
            export PIC_BACKEND="cuda"
            . /etc/profile
            module load gcc/5.5.0 boost/1.65.1 cmake/3.15.0 cuda/9.2.148 openmpi/3.0.4
            module load libSplash/1.7.0 adios/1.13.1
            module load pngwriter/0.7.0 zlib/1.2.11
            module load libjpeg-turbo/1.5.1 icet/2.1.1 jansson/2.9 isaac/1.4.0

            # compile all examples, fetch output and return code
            $cnf_gitdir/bin/pic-compile -l -q -j $cnf_numParallel \
                                $cnf_gitdir/share/picongpu/examples $cnf_builddir \
                                &> $cnf_builddir/outputColored

            echo $? > $cnf_builddir"/returnCode"

            # add information to the head of the output
            cp $cnf_builddir"/outputColored" $cnf_builddir"/outputColored_short"
            echo "" > $cnf_builddir"/outputColored"
            if [ $cnf_numParallel -gt 1 ] ; then
                for i in `ls $cnf_builddir"/build/"`
                do
                    returnCode=`cat $cnf_builddir"/build/"$i"/returnCode"`
                    if [ "$returnCode" != "0" ] ; then
                        cat $cnf_builddir"/build/"$i"/compile.log" >> $cnf_builddir"/outputColored"
                    fi
                done
            fi
            cat $cnf_builddir"/outputColored_short" >> $cnf_builddir"/outputColored"

            # format output
            cat $cnf_builddir"/outputColored" | \
              sed -r "s/\x1B\[([0-9]{1,2}(;[0-9]{1,2})?)?[m|K]//g" \
              > $cnf_builddir"/output"

            echo "Error Analysis:"

            # analyse output
            returnCode=`cat $cnf_builddir"/returnCode"`
            echo "Compile Suite return code: $returnCode"

            # is +/- integer?
            if ! [[ "$returnCode" =~ ^[\-0-9]+$ ]] ; then
                returnCode=1
                state=0 # suite errored (no return code)
                echo "NO return code! (Suite errored)"
            else
                if [ "$returnCode" -eq "0" ] ; then
                    echo "All right :)"
                    state=1
                else
                    echo "Non-Zero return code!"
                    state=-1
                fi
            fi

            # create conclusion, update status (and send mails)
            conclusion "$state" "$lastUser" "$lastUserMail" "$sha" "$eventid" "$logEntry" "$cnf_builddir""/output"

        fi
    #done

#done

# del guard
rm "$thisDir"runGuard
