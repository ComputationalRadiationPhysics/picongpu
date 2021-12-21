#!/bin/bash -l
#

# output directory from startup arguments
output_dir=${1:-"/tmp/khi8_001/"}

if [ "$output_dir" = "-h" ] || [ "$output_dir" = "--help" ]
then
  echo "Usage:"
  echo "  $0 [output_directory]"
fi

echo ""
echo "Running KHI Benchmark on 8 GPUs..."
echo ""


# start PIConGPU
cd /opt/picInputs/khi
/usr/bin/time -f "%e" tbg \
  -f \
  -s "bash -l" \
  -c etc/picongpu/8_bench.cfg \
  -t etc/picongpu/bash/mpirun.tpl \
  $output_dir

echo ""
echo "Simulation finished! See the created output in:"
echo "    $output_dir"
echo ""
