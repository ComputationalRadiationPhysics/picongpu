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

# global configs ##############################################################
#
cnf_scheduler_secret="..."
cnf_scheduler="https://example.com?client="$cnf_scheduler_secret

cnf_gitdir="$HOME/pic_git/"
cnf_exportdir="$HOME/pic_export/"
cnf_extfile="$HOME/machines/exchange.ext2"

cnf_imgSrc="$HOME/.aqemu/Debian7_Cuda4_2_HDA"
cnf_imgClone="$HOME/machines/Debian7_Cuda4_2_HDA"

cnf_statsfile="stats.cnf"

# number of tests to compile in parallel
# = cpu's of the virtual machine
cnf_numParallel=2

# send a congrats mail if N commits in a row where successful
# (kind of a heart beat for this script)
cnf_congrats=10

# mail settings
cnf_smtp="smtp.example.com:25"
cnf_smtp_auth_user=""
cnf_smtp_auth_password=""
#cnf_sendcharsets=
cnf_from="someone@example.com"

cnf_rctp_to="someoneelse@example.com"

# preview N lines of the compile output for failing examples
cnf_mail_preview=60

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
