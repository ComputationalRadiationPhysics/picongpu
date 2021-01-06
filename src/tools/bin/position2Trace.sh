#!/usr/bin/env bash
#
# Copyright 2013-2021 Rene Widera, Richard Pausch
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

#$1 picongpu file with output
#$2 destination file

if [ $# -ne 2 ] ; then
    exit
fi

grep e_position $1 | awk '{print($6"\t"$7"\t"$8"\t"$9"\t"$10"\t"$11"\t"$12)}' | sed 's/,/\t/g' | tr -d "{" | tr -d "}" > $2
