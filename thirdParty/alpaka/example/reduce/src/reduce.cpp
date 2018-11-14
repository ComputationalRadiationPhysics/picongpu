/**
 * \file
 * Copyright 2018 Jonas Schenke, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * alpaka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * alpaka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with alpaka.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "alpakaConfig.hpp"
#include "kernel.hpp"
#include <alpaka/alpaka.hpp>
#include <cstdlib>
#include <iostream>

// hardcode the serial CPU accelerator
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

// use defines of a specific accelerator
using Accelerator = CpuSerial;

using DevAcc = Accelerator::DevAcc;
using DevHost = Accelerator::DevHost;
using QueueAcc = Accelerator::Stream;
using Acc = Accelerator::Acc;
using PltfAcc = Accelerator::PltfAcc;
using PltfHost = Accelerator::PltfHost;
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
template<typename T, typename TFunc>
T reduce(DevHost devHost, DevAcc devAcc, QueueAcc queue, uint64_t n, alpaka::mem::buf::Buf<DevHost, T, Dim, Idx> hostMemory, TFunc func)
{
    static constexpr uint64_t blockSize = getMaxBlockSize<Accelerator, 256>();

    // calculate optimal block size (8 times the MP count proved to be
    // relatively near to peak performance in benchmarks)
    uint32_t blockCount = static_cast<uint32_t>(
        alpaka::acc::getAccDevProps<Acc, DevAcc>(devAcc).m_multiProcessorCount *
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

    // execute first kernel
    auto const exec1(alpaka::kernel::createTaskExec<Acc>(
        workDiv1,
        kernel1,
        alpaka::mem::view::getPtrNative(sourceDeviceMemory),
        alpaka::mem::view::getPtrNative(destinationDeviceMemory),
        n,
        func));

    // reduce the last block
    auto const exec2(alpaka::kernel::createTaskExec<Acc>(
        workDiv2,
        kernel2,
        alpaka::mem::view::getPtrNative(destinationDeviceMemory),
        alpaka::mem::view::getPtrNative(destinationDeviceMemory),
        blockCount,
        func));

    // enqueue both kernels
    alpaka::queue::enqueue(queue, exec1);
    alpaka::queue::enqueue(queue, exec2);

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
    
    DevAcc devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(dev));
    DevHost devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));
    QueueAcc queue(devAcc);

    // calculate optimal block size (8 times the MP count proved to be
    // relatively near to peak performance in benchmarks)
    uint32_t blockCount = static_cast<uint32_t>(
        alpaka::acc::getAccDevProps<Acc, DevAcc>(devAcc).m_multiProcessorCount *
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
