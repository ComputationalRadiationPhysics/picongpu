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

# Contact the scheduler and request a new task to work on
# create a read only, clean source
# image

thisDir=$(cd `dirname $0` && pwd)"/"

. "$thisDir"config.sh

security_check $thisDir

# clean up old stuff
#
rm -rf $cnf_gitdir/*
mkdir -p $cnf_gitdir
cd $cnf_gitdir
rm -rf $cnf_exportdir
mkdir -p $cnf_exportdir

# check for work
#
sched=`curl -d'payload={"action":"request"}' $cnf_scheduler 2>/dev/null`
if [ $? -ne 0 ]; then
    echo "Error contacting scheduler at $cnf_scheduler"
    exit 1
fi

workType=`echo -e "$sched" | grep "etype" | awk -F':' '{print $2}' | awk -F'"' '{print $2}'`

# HEAD commit properties
eventid=`echo -e "$sched" | grep "id" | head -n1 | awk -F'":' '{print $2}' | awk -F'"' '{print $2}'`
owner=`echo -e "$sched" | grep "owner" | head -n1 | awk -F'":' '{print $2}' | awk -F'"' '{print $2}'`
repo=`echo -e "$sched" | grep "repo" | head -n1 | awk -F'":' '{print $2}' | awk -F'"' '{print $2}'`
git=`echo -e "$sched" | grep "git" | head -n1 | awk -F'":' '{print $2}' | awk -F'"' '{print $2}' | sed 's|\\\/|\/|g'`
sha=`echo -e "$sched" | grep "sha" | head -n1 | awk -F'":' '{print $2}' | awk -F'"' '{print $2}'`

echo $eventid > "$thisDir"runGuard

if [ "$workType" == "commit" ]
then
    git clone $git .
    git checkout $sha
elif [ "$workType" == "pull" ]
then
    # BASE repos commit properties
    owner_b=`echo -e "$sched" | grep "owner" | tail -n1 | awk -F'":' '{print $2}' | awk -F'"' '{print $2}'`
    repo_b=`echo -e "$sched" | grep "repo" | tail -n1 | awk -F'":' '{print $2}' | awk -F'"' '{print $2}'`
    git_b=`echo -e "$sched" | grep "git" | tail -n1 | awk -F'":' '{print $2}' | awk -F'"' '{print $2}' | sed 's|\\\/|\/|g'`
    sha_b=`echo -e "$sched" | grep "sha" | tail -n1 | awk -F'":' '{print $2}' | awk -F'"' '{print $2}'`

    git clone $git_b .
    git checkout -b mergeTest $sha_b
    git pull $git $sha
    if [ $? -ne 0 ] ; then
        # merge failed
        cd -
        exit 2
    fi
else
    cd -
    exit 1
fi

rsync -a --exclude=.git . $cnf_exportdir

# iso
#genisoimage -D -iso-level 4 -quiet \
#  -input-charset=iso8859-1 \
#  -o $cnf_isofile $cnf_exportdir

# raw (loop dev)
rm -rf $cnf_extfile
qemu-img create -f raw $cnf_extfile 1G
/sbin/mke2fs -q -F $cnf_extfile

mntdir=$thisDir"tmp_mount"
mkdir $mntdir
fuseext2 -o rw+ $cnf_extfile $mntdir

mkdir $mntdir/pic_export
cp $thisDir"compileRun.sh" $mntdir/
cp -R $cnf_exportdir/* $mntdir/pic_export

# unmount
sync
fusermount -u $mntdir
rm -rf $mntdir

# go back
cd -
