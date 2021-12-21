#!/usr/bin/env bash
#
# Copyright 2017-2021 Rene Widera, Alexander Matthes
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
# OMP_NUM_THREADS is set to the number of threads within the numa node at
# default. If not every hardware thread of a core shall be used, the environment
# variable NUMA_HW_THREADS_PER_PHYSICAL_CORE can be steered to change
# OMP_NUM_THREADS accordingly. `numactl` and `/proc/cpuinfo` are used to
# calculate the number of numa nodes, sockets, and number of cores and hardware
# threads per numa node or socket.
#
# NUMA_HW_THREADS_PER_PHYSICAL_CORE is the number of used hardware threads
# (read "hyperthreads" for Intel) per physical core. If hyperthreading shall not
# be used, use "export NUMA_HW_THREADS_PER_PHYSICAL_CORE=1" before using
# cpuNumaStarter.sh
#
# dependencies: numactl, openmpi, hwloc or /proc/cpuinfo, whereby hwloc is more
# accurate

numactl --show &>/dev/null

if [ $? -eq 0 ] ; then
    numNumaNodes=`numactl --show | grep nodebind | awk '{print $NF}'`
    let numNumaNodes=numNumaNodes+1

    numHardwareThreadsTotal=`hwloc-info | grep "PU (type #6)" | awk '{print $3}'`
    if [ ! -n "$numHardwareThreadsTotal" ] ; then
        numHardwareThreadsTotal=`cat /proc/cpuinfo | grep "processor" | sort -r -V | head -n 1 | awk '{print $NF}'`
        let numHardwareThreadsTotal=numHardwareThreadsTotal+1
    fi

    numCoresTotal=`hwloc-info | grep "Core (type #5)" | awk '{print $3}'`
    if [ ! -n "$numCoresTotal" ] ; then
        # For some special cases like AMD Bulldozer on multiple sockets, this
        # work around with /proc/cpuinfo will result in a too small number of
        # cores as the packages are not ecognized correctly.
        numCoresPerSocket=`cat /proc/cpuinfo | grep "cpu cores" | head -n 1 | awk '{print $NF}'`
        numSockets=`cat /proc/cpuinfo | grep "physical id" | sort -r -V | head -n 1 | awk '{print $NF}'`
        let numSockets=numSockets+1
        numCoresTotal="$(( numCoresPerSocket * numSockets ))"
    fi

    numCoresPerNumaNode="$(( numCoresTotal / numNumaNodes ))"

    numHardwareThreadsPerCore="$(( numHardwareThreadsTotal / numCoresTotal ))"
    if [ ! -n "$NUMA_HW_THREADS_PER_PHYSICAL_CORE" ] ; then
        NUMA_HW_THREADS_PER_PHYSICAL_CORE="$(( numHardwareThreadsPerCore ))"
    fi

    useOpenMPNumThreads="$(( NUMA_HW_THREADS_PER_PHYSICAL_CORE * numCoresPerNumaNode ))"

    if [ -n "$OMPI_COMM_WORLD_LOCAL_RANK" ] ; then
        let localRank=$OMPI_COMM_WORLD_LOCAL_RANK
    elif [ -n "$PMI_RANK" ] ; then
        let localRank=$PMI_RANK
    fi

    if [ -n "$localRank" ] ; then
        let useNumaNode=localRank%numNumaNodes

        export MPI_LOCAL_RANK=$localRank
        export OMP_NUM_THREADS=$useOpenMPNumThreads

        numactl --cpunodebind="$useNumaNode" --preferred="$useNumaNode" $*
    else
        echo "WARNING: OpenMPI missing, start without thread pinning" >&2
        $*
    fi
else
    echo "WARNING: numactl not found, start without thread pinning" >&2
    $*
fi

exit $?
