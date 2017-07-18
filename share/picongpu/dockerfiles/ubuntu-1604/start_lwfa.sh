#!/bin/bash
#

isaac &

echo ""
echo "Let's watch a laser-plasma movie!"
echo "  http://plasma.ninja/isaac_1_3_0/interface.htm"
echo ""

# wait until server is up
sleep 5

# start PIConGPU
cd paramSets/lwfa
tbg \
  -s bash \
  -c etc/picongpu/0001gpus_isaac.cfg \
  -t etc/bash/bash_mpirun.tpl \
  /tmp/lwfa_001
