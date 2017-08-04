#!/bin/bash

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

        echo "localRank=$localRank cpuSocket=$cpuSocket numCores=$numCores numCPUSockets=$numCPUSockets numCoresPerSocket=$numCoresPerSocket"

        numactl --cpunodebind="$cpuSocket" --preferred="$cpuSocket" $*
    else
        echo "ERROR: OpenMPI missing, start without thread pinning" >&2
        $*
    fi
else
    echo "ERROR: numactl not found, start without thread pinning" >&2
    $*
fi

exit $?
