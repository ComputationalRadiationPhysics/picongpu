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

# global configs ##############################################################
#
cnf_scheduler_secret="..."
cnf_scheduler="https://example.com?client="$cnf_scheduler_secret

# temporary git & build directory: careful, will be purged!
cnf_gitdir="$HOME/picongpu-src/"
cnf_builddir="$HOME/build/"

# number of tests to compile in parallel
cnf_numParallel=16

# preview N lines of the compile output for failing examples
cnf_mail_preview=120

# global functions ############################################################
#

# security check
#
function security_check()
{
    if [ -d "$1/.svn" ] || [ -d "$1/.git" ]; then
        echo "SECURITY RISK: Do not use this script"
        echo "within a repository! (Do not update"
        echo "it automatically!)"
        exit 1
    fi
}
