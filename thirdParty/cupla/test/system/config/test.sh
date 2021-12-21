#!/usr/bin/env bash
set -eu -o pipefail

# test compile and execution of a cupla program using its header-only variant
# $1 C++ compiler name

test_code_dir=$(dirname $0)

# compiles the test case for the given accelerator
# $1 flag if the compiled binary should be executed, 1 == execute binary else do not test binary 
# $2 compiler name
# $3 accelerator name (CpuOmp2Blocks, CpuOmp2Threads, CpuSerial, CpuThreads, GpuCudaRt, GpuHipRt)
# $4 additional compiler flags
function compile {
    execute_bin=$1
    compiler_name=$2
    acc_name=$3
    if [ $# -eq 4 ] ; then
        compiler_flags="$4"
    fi
    echo "execute: "${compiler_name} ${test_code_dir}/main.cpp ${test_code_dir}/kernel.cpp  -I${test_code_dir}/../../../include -I${test_code_dir}/../../../alpaka/include -std=c++14 -DCUPLA_ACC_${acc_name} -DCUPLA_HEADER_ONLY -o ${acc_name} ${compiler_flags}
    ret=$(${compiler_name} ${test_code_dir}/main.cpp ${test_code_dir}/kernel.cpp  -I${test_code_dir}/../../../include -I${test_code_dir}/../../../alpaka/include -std=c++14 -DCUPLA_ACC_${acc_name} -DCUPLA_HEADER_ONLY -o ${acc_name} ${compiler_flags} && \
        { echo 0; } || { echo 1; })
    if [ $ret -eq 0 ] && [ $execute_bin -eq 1 ] ; then
        ret=$(./$acc_name && { echo 0; } || { echo 1; })
    fi
    if [ $ret -ne 0 ] ; then
        echo "Config header test failed for accelerator '$acc_name' with compiler flags '$compiler_flags'" >&2
    fi
    return $ret
}

boost_include="-I$BOOST_ROOT/include"

compile 1 "$1" CpuOmp2Blocks "-fopenmp $boost_include"
compile 1 "$1" CpuOmp2Blocks "-fopenmp -DCUPLA_STREAM_ASYNC_ENABLE=1 $boost_include"
compile 0 "$1" CpuOmp2Threads "-fopenmp $boost_include"
compile 0 "$1" CpuOmp2Threads "-fopenmp -DCUPLA_STREAM_ASYNC_ENABLE=1 $boost_include"
# -pthread and -lpthread  is required for std::future in C++11
compile 1 "$1" CpuSerial "-pthread -lpthread  $boost_include"
compile 1 "$1" CpuSerial "-pthread -pthread -lpthread -DCUPLA_STREAM_ASYNC_ENABLE=1 $boost_include"
compile 0 "$1" CpuThreads "-pthread -lpthread $boost_include"
compile 0 "$1" CpuThreads "-pthread -lpthread -DCUPLA_STREAM_ASYNC_ENABLE=1 $boost_include"

nvcc_found=$(which nvcc >/dev/null && { echo 0; } || { echo 1; })
if [ $nvcc_found -eq 0 ] ; then
    compile 0 "nvcc -x cu" GpuCudaRt "-DALPAKA_ACC_GPU_CUDA_ENABLE=ON -DALPAKA_ACC_GPU_CUDA_ONLY_MODE=ON $boost_include --expt-relaxed-constexpr"
    compile 0 "nvcc -x cu" GpuCudaRt "-DALPAKA_ACC_GPU_CUDA_ENABLE=ON -DALPAKA_ACC_GPU_CUDA_ONLY_MODE=ON -DCUPLA_STREAM_ASYNC_ENABLE=1 $boost_include --expt-relaxed-constexpr"
else 
    echo "skip GpuCudaRt: nvcc not found" >&2
fi

hipcc_found=$(which hipcc >/dev/null && { echo 0; } || { echo 1; })
if [ $hipcc_found -eq 0 ] ; then
    rocm_root="/opt/rocm"
    hip_include="-I${rocm_root} -I${rocm_root}/rocrand/hiprand/include/ -I${rocm_root}/rocrand/include/"
    compile 1 "hipcc" GpuHipRt "-DALPAKA_ACC_GPU_HIP_ENABLE=ON -DALPAKA_ACC_GPU_HIP_ONLY_MODE=ON $hip_include $boost_include"
    compile 1 "hipcc" GpuHipRt "-DALPAKA_ACC_GPU_HIP_ENABLE=ON -DALPAKA_ACC_GPU_HIP_ONLY_MODE=ON -lpthread -DCUPLA_STREAM_ASYNC_ENABLE=1 $hip_include $boost_include"
else 
    echo "skip GpuHipRt:  hipcc not found" >&2
fi
