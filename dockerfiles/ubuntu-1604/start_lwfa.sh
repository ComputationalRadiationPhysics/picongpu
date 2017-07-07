#!/bin/bash
#

isaac &

mpirun -n 1 paramSets/lwfa/bin/picongpu \
    -d 1 1 1 \
    -g 32 32 32 \
    -s 1000 \
    --softRestarts 10 \
    --isaac.period 1
