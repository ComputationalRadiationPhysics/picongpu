### BEGIN INIT INFO
# Provides:          compileRun.sh
# Required-Start:    $all
# Required-Stop:
# Default-Start:     2 3 4 5
# Default-Stop:      0 1 6
# Short-Description: Run-Compile Script for PIConGPU
# Description:       This script starts the compile
#                    suite for PIConGPU with user rights.
#                    The output and result are dumped
#                    afterwards and the machine will
#                    shutdown.
### END INIT INFO

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

user="picman"
numParallel=2

exch="/home/"$user"/exchange"
picExport=$exch"/pic_export"
buildDir="/home/"$user"/tmp_build"

case "$1" in
    start)
	$0 "asyncStart" > $exch"/cron.log" 2>&1 &
	;;
    asyncStart)
        # clean tmp build folder
        rm -rf $buildDir
        mkdir -p $buildDir
        chown -R $user:$user $buildDir
        chown -R $user:$user $exch

        # set some env, like nvcc alias
        exec_user=". ~/.login &&"
        # disable png and HDF5
        #exec_user="$exec_user export PIC_COMPILE_SUITE_CMAKE='-DPIC_ENABLE_PNG=OFF"
        #exec_user="$exec_user -DPIC_ENABLE_HDF5=OFF -DCUDA_ARCH=sm_20' &&"
        # compile
        exec_user="$exec_user $picExport/compile -l -q -j $numParallel $picExport/examples $buildDir "
        #exec_user="$exec_user $picExport/compile -q $picExport/examples/TermalTest $buildDir "
        # fetch output and return code
        exec_user="$exec_user &> $exch/outputColored"

        # compile tests as user $user
        # (note: $user can use passwordless sudo)
        ssh -i "/home/"$user"/.ssh/id_rsa" "$user"@127.0.0.1 "$exec_user"
        echo $? > $exch"/returnCode"

        # add information to the head of the output
        cp $exch"/outputColored" $exch"/outputColored_short"
        echo "" > $exch"/outputColored"
        if [ $numParallel -gt 1 ] ; then
            for i in `ls $buildDir"/build/"`
            do
                returnCode=`cat $buildDir"/build/"$i"/returnCode"`
                if [ "$returnCode" != "0" ] ; then
                    cat $buildDir"/build/"$i"/compile.log" >> $exch"/outputColored"
                fi
            done
        fi
        cat $exch"/outputColored_short" >> $exch"/outputColored"

        # format output
        cat $exch"/outputColored" | \
          sed -r "s/\x1B\[([0-9]{1,2}(;[0-9]{1,2})?)?[m|K]//g" \
          > $exch"/output"

        rm -rf $exch"/outputColored.gz"
        gzip $exch"/outputColored"

        # clean tmp build folder
        rm -rf $buildDir

        chmod a+rwx -R $exch
        sync

        # shutdown
        shutdown -h now
        ;;
    stop)
        echo "$0: nothing to do..."
        ;;
    restart)
        echo "$0: nothing to do..."
        ;;
esac

exit 0
