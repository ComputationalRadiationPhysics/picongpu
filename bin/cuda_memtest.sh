#!/usr/bin/env bash
#
# Copyright 2013-2021 Rene Widera
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

old_path=`pwd`

cd `dirname $0`

# add local folder to binary search path
export PATH=".:$PATH"

#activate tests for cuda_memtest see: ./cuda_memtest --list_tests
enable_gpu_tests="--enable_test 2 --enable_test 4"


if [ ! -x "./mpiInfo" ] ; then
   echo "file ./mpiInfo not exists or is not executable" >&2
   exit 1
fi
host_rank=`mpiInfo --mpi_host_rank | grep mpi_host_rank | cut -d":" -f2 | tr -d " "`
output=`cuda_memtest --disable_all --device $host_rank $enable_gpu_tests --num_passes 1 --exit_on_error`

if [ $? -ne 0 ] ; then
   host_name=`hostname`
   echo -e "$output" > $old_path/cuda_memtest_"$host_name"_"$host_rank".err
   echo cuda_memtest crash: see file $old_path/cuda_memtest_"$host_name"_"$host_rank".err >&2
   exit 1
fi
exit 0
