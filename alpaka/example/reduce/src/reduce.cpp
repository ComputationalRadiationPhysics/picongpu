/* Copyright 2019 Benjamin Worpitz, Jonas Schenke, Matthias Werner
 *
 * This file exemplifies usage of alpaka.
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED “AS IS” AND ISC DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL ISC BE LIABLE FOR ANY
 * SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
 * IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include "alpakaConfig.hpp"
#include "kernel.hpp"
#include <alpaka/alpaka.hpp>
#include <cstdlib>
#include <iostream>

// It requires support for extended lambdas when using nvcc as CUDA compiler.
// Requires sequential backend if CI is used
#if (!defined(__NVCC__) || (defined(__NVCC__) && defined(__CUDACC_EXTENDED_LAMBDA__) )) && \
    (!defined(ALPAKA_CI) || defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED))

// use defines of a specific accelerator from alpakaConfig.hpp
// that are defined in alpakaConfig.hpp
// - GpuCudaRt
// - CpuThreads
// - CpuOmp2Blocks
// - CpuOmp4
// - CpuSerial
//
using Accelerator = CpuSerial;

using Acc = Accelerator::Acc;
using Host = Accelerator::Host;
using QueueProperty = alpaka::queue::Blocking;
using QueueAcc = alpaka::queue::Queue<Acc, QueueProperty>;
using MaxBlockSize = Accelerator::MaxBlockSize;

//-----------------------------------------------------------------------------
//! Reduces the numbers 1 to n.
//!
//! \tparam T The data type.
//! \tparam TFunc The data type of the reduction functor.
//!
//! \param devHost The host device.
//! \param devAcc The accelerator object.
//! \param queue The device queue.
//! \param n The problem size.
//! \param hostMemory The buffer containing the input data.
//! \param func The reduction function.
//!
//! Returns true if the reduction was correct and false otherwise.
template<typename T, typename DevHost, typename DevAcc, typename TFunc>
T reduce(DevHost devHost, DevAcc devAcc, QueueAcc queue, uint64_t n, alpaka::mem::buf::Buf<DevHost, T, Dim, Idx> hostMemory, TFunc func)
{
    static constexpr uint64_t blockSize = getMaxBlockSize<Accelerator, 256>();

    // calculate optimal block size (8 times the MP count proved to be
    // relatively near to peak performance in benchmarks)
    uint32_t blockCount = static_cast<uint32_t>(
        alpaka::acc::getAccDevProps<Acc>(devAcc).m_multiProcessorCount *
        8);
    uint32_t maxBlockCount = static_cast<uint32_t>(
        (((n + 1) / 2) - 1) / blockSize + 1); // ceil(ceil(n/2.0)/blockSize)

    if (blockCount > maxBlockCount)
        blockCount = maxBlockCount;

    alpaka::mem::buf::Buf<DevAcc, T, Dim, Extent> sourceDeviceMemory =
        alpaka::mem::buf::alloc<T, Idx>(devAcc, n);

    alpaka::mem::buf::Buf<DevAcc, T, Dim, Extent> destinationDeviceMemory =
        alpaka::mem::buf::alloc<T, Idx>(
            devAcc, static_cast<Extent>(blockCount));

    // copy the data to the GPU
    alpaka::mem::view::copy(queue, sourceDeviceMemory, hostMemory, n);

    // create kernels with their workdivs
    ReduceKernel<blockSize, T, TFunc> kernel1, kernel2;
    WorkDiv workDiv1{ static_cast<Extent>(blockCount),
                      static_cast<Extent>(blockSize),
                      static_cast<Extent>(1) };
    WorkDiv workDiv2{ static_cast<Extent>(1),
                      static_cast<Extent>(blockSize),
                      static_cast<Extent>(1) };

    // create main reduction kernel execution task
    auto const taskKernelReduceMain(alpaka::kernel::createTaskKernel<Acc>(
        workDiv1,
        kernel1,
        alpaka::mem::view::getPtrNative(sourceDeviceMemory),
        alpaka::mem::view::getPtrNative(destinationDeviceMemory),
        n,
        func));

    // create last block reduction kernel execution task
    auto const taskKernelReduceLastBlock(alpaka::kernel::createTaskKernel<Acc>(
        workDiv2,
        kernel2,
        alpaka::mem::view::getPtrNative(destinationDeviceMemory),
        alpaka::mem::view::getPtrNative(destinationDeviceMemory),
        blockCount,
        func));

    // enqueue both kernel execution tasks
    alpaka::queue::enqueue(queue, taskKernelReduceMain);
    alpaka::queue::enqueue(queue, taskKernelReduceLastBlock);

    //  download result from GPU
    T resultGpuHost;
    auto resultGpuDevice =
        alpaka::mem::view::ViewPlainPtr<DevHost, T, Dim, Idx>(
            &resultGpuHost, devHost, static_cast<Extent>(blockSize));

    alpaka::mem::view::copy(queue, resultGpuDevice, destinationDeviceMemory, 1);

    return resultGpuHost;
}

int main()
{
    // select device and problem size
    const int dev = 0;
    uint64_t n = 1 << 28;

    using T = uint32_t;
    static constexpr uint64_t blockSize = getMaxBlockSize<Accelerator, 256>();

    auto devAcc = alpaka::pltf::getDevByIdx<Acc>(dev);
    auto devHost = alpaka::pltf::getDevByIdx<Host>(0u);
    QueueAcc queue(devAcc);

    // calculate optimal block size (8 times the MP count proved to be
    // relatively near to peak performance in benchmarks)
    uint32_t blockCount = static_cast<uint32_t>(
        alpaka::acc::getAccDevProps<Acc>(devAcc).m_multiProcessorCount *
        8);
    uint32_t maxBlockCount = static_cast<uint32_t>(
        (((n + 1) / 2) - 1) / blockSize + 1); // ceil(ceil(n/2.0)/blockSize)

    if (blockCount > maxBlockCount)
        blockCount = maxBlockCount;

    // allocate memory
    auto hostMemory = alpaka::mem::buf::alloc<T, Idx>(devHost, n);

    T *nativeHostMemory = alpaka::mem::view::getPtrNative(hostMemory);

    // fill array with data
    for (uint64_t i = 0; i < n; i++)
        nativeHostMemory[i] = static_cast<T>(i + 1);

    // define the reduction function
    auto addFn = [] ALPAKA_FN_ACC(T a, T b) -> T { return a + b; };

    // reduce
    T result = reduce<T>(devHost, devAcc, queue, n, hostMemory, addFn);
    alpaka::wait::wait(queue);

    // check result
    T expectedResult = static_cast<T>(n / 2 * (n + 1));
    if (result != expectedResult)
    {
        std::cerr << "Results don't match: " << result << " != " << expectedResult
                  << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "Results match.\n";

    return EXIT_SUCCESS;
}

#else

int main() {
    return EXIT_SUCCESS;
}

#endif
