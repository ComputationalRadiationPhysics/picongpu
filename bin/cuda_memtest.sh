#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Rene Widera
#
# SPDX-License-Identifier: GPL-3.0-or-later
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
output=`cuda_memtest --disable_all --device $host_rank $enable_gpu_tests --num_passes 1 --exit_on_error 2>&1`

if [ $? -ne 0 ] ; then
   host_name=`hostname`
   if [ ! -d "$old_path" ]; then
       echo "Error: $0 did not find directory: $old_path (on host: $host_name with rank: $host_rank)" >&2
       echo "error message of memtest is:" >&2
       echo -e "$output" >&2
       exit 2
   else
      echo -e "$output" > $old_path/cuda_memtest_"$host_name"_"$host_rank".err
      echo cuda_memtest crash: see file $old_path/cuda_memtest_"$host_name"_"$host_rank".err >&2
      exit 1
   fi
fi
exit 0
