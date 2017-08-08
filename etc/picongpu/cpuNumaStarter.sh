#!/usr/bin/env bash
#
# Copyright 2017 Rene Widera
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

# This tool binds a process and main memory to a numa node.
# OMP_NUM_THREADS is set to the number of threads within the numa node.
# `numactl` is used to calculate the number and cpu's per numa node.
#
# dependencies: numctl, openmpi

numactl --show &>/dev/null

if [ $? -eq 0 ] ; then
    numCores=`numactl --show | grep physcpubind | awk '{print $NF}'`
    let numCores=numCores+1
    numCPUSockets=`numactl --show | grep nodebind | awk '{print $NF}'`
    let numCPUSockets=numCPUSockets+1

    let numCoresPerSocket=numCores/numCPUSockets

    if [ -n "$OMPI_COMM_WORLD_LOCAL_RANK" ] ; then

        let localRank=$OMPI_COMM_WORLD_LOCAL_RANK
        let cpuSocket=localRank%numCPUSockets

        export MPI_LOCAL_RANK=$localRank
        export OMP_NUM_THREADS=$numCoresPerSocket

        numactl --cpunodebind="$cpuSocket" --preferred="$cpuSocket" $*
    else
        echo "WARNING: OpenMPI missing, start without thread pinning" >&2
        $*
    fi
else
    echo "WARNING: numactl not found, start without thread pinning" >&2
    $*
fi

exit $?
