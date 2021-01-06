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

# Contact the scheduler and request a new task to work on
# create a read only, clean source
# image

thisDir=$(cd `dirname $0` && pwd)"/"

. "$thisDir"config.sh

security_check $thisDir

# clean up old stuff
#
rm -rf $cnf_gitdir
mkdir -p $cnf_gitdir
cd $cnf_gitdir

# check for work
#
sched=`curl -d'payload={"action":"request"}' $cnf_scheduler 2>/dev/null`
if [ $? -ne 0 ]; then
    echo "Error contacting scheduler at $cnf_scheduler"
    exit 1
fi

# decode JSON escaped strings
#
sched=${sched//\\\\/\\} # \
sched=${sched//\\\//\/} # /
sched=${sched//\\\'/\'} # '
sched=${sched//\\\"/\"} # "
sched=${sched//\\\t/	} # \t
sched=${sched//\\\n/
} # \n
sched=${sched//\\\r/^M} # \r
sched=${sched//\\\f/^L} # \f
sched=${sched//\\\b/^H} # \b

workType=`echo -e "$sched" | grep '"etype"' | awk -F':' '{print $2}' | awk -F'"' '{print $2}'`

# HEAD commit properties
eventid=`echo -e "$sched" | grep '"id"' | head -n1 | awk -F'":' '{print $2}' | awk -F',' '{print $1}'`
owner=`echo -e "$sched" | grep '"owner"' | head -n1 | awk -F'":' '{print $2}' | awk -F'"' '{print $2}'`
repo=`echo -e "$sched" | grep '"repo"' | head -n1 | awk -F'":' '{print $2}' | awk -F'"' '{print $2}'`
git=`echo -e "$sched" | grep '"git"' | head -n1 | awk -F'":' '{print $2}' | awk -F'"' '{print $2}'`
sha=`echo -e "$sched" | grep '"sha"' | head -n1 | awk -F'":' '{print $2}' | awk -F'"' '{print $2}'`

echo $eventid > "$thisDir"runGuard

if [ "$workType" == "commit" ]
then
    git clone -q $git .
    if [ $? -ne 0 ] ; then
        echo "git clone failed"
        exit 2
    fi
    git checkout -q $sha
elif [ "$workType" == "pull" ]
then
    # BASE repos commit properties
    owner_b=`echo -e "$sched" | grep '"owner"' | tail -n1 | awk -F'":' '{print $2}' | awk -F'"' '{print $2}'`
    repo_b=`echo -e "$sched" | grep '"repo"' | tail -n1 | awk -F'":' '{print $2}' | awk -F'"' '{print $2}'`
    git_b=`echo -e "$sched" | grep '"git"' | tail -n1 | awk -F'":' '{print $2}' | awk -F'"' '{print $2}'`
    sha_b=`echo -e "$sched" | grep '"sha"' | tail -n1 | awk -F'":' '{print $2}' | awk -F'"' '{print $2}'`

    branch=`echo -e "$sched" | grep '"branch"' | head -n1 | awk -F'":' '{print $2}' | awk -F'"' '{print $2}'`

    git clone -q $git_b .
    if [ $? -ne 0 ] ; then
        echo "git clone failed"
        exit 2
    fi
    # simulate a merge of the commit on the base branch
    git checkout -q -b mergeTest $sha_b
    git remote add -f pull_repo $git > /dev/null 2>&1
    git merge -q $sha
    if [ $? -ne 0 ] ; then
        echo "git merge failed"
        exit 2
    fi
else
    exit 1
fi
