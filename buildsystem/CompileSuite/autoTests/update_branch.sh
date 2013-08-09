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
# Update a branch from $1 and
# create a read only, clean source
# image

thisDir=$(cd `dirname $0` && pwd)"/"

. "$thisDir"config.sh

# options
#
branch=${1:-"branches/dev/"}
rev=${2:-"HEAD"}

security_check $thisDir

# output options
#
echo "Branch: $branch"
echo "Revision: $rev"

# execute
#
cd $cnf_svndir$branch
svn update -r $rev
if [ $? -ne 0 ]; then
#    echo "Version $rev in $branch does not exist."
    exit 1
fi

rm -rf $cnf_exportdir
mkdir -p $cnf_exportdir
svn export --force . $cnf_exportdir

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
