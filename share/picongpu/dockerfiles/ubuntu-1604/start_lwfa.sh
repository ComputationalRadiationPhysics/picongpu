#!/bin/bash -l
#

isaac &
server_id=$!

echo ""
echo "Let's watch a laser-plasma movie!"
echo "  http://plasma.ninja/isaac_1_3_0/interface.htm"
echo ""

# wait until server is up
sleep 5

# start PIConGPU
cd /opt/picInputs/lwfa
tbg \
  -s "bash -l" \
  -c etc/picongpu/0001gpus_isaac.cfg \
  -t etc/picongpu/bash/mpirun.tpl \
  /tmp/lwfa_001

# kill the isaac server after tbg returns
kill $server_id
