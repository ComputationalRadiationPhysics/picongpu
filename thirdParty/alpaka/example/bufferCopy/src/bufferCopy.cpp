/**
 * \file
 * Copyright 2014-2018 Erik Zenker, Benjamin Worpitz
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

#include <alpaka/alpaka.hpp>

#include <iostream>
#include <cstdint>

//-----------------------------------------------------------------------------
template <size_t width>
ALPAKA_FN_ACC size_t linIdxToPitchedIdx(size_t const globalIdx, size_t const pitch)
{
    const size_t idx_x = globalIdx % width;
    const size_t idx_y = globalIdx / width;
    return idx_x + idx_y * pitch;
}

//#############################################################################
//! Prints all elements of the buffer.
struct PrintBufferKernel
{
    //-----------------------------------------------------------------------------
    template<
        typename TAcc,
        typename TData,
        typename TExtent>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        TData const * const buffer,
        TExtent const & extents,
        size_t const pitch) const
    -> void
    {
        auto const globalThreadIdx = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent = alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearizedGlobalThreadIdx = alpaka::idx::mapIdx<1u>(
            globalThreadIdx,
            globalThreadExtent);

        for(size_t i(linearizedGlobalThreadIdx[0]); i < extents.prod(); i += globalThreadExtent.prod())
        {
            // NOTE: hard-coded for unsigned int
            printf("%u:%u ", static_cast<uint32_t>(i), static_cast<uint32_t>(buffer[linIdxToPitchedIdx<2>(i,pitch)]));
        }
    }
};


//#############################################################################
//! Tests if the value of the buffer on index i is equal to i.
struct TestBufferKernel
{
    //-----------------------------------------------------------------------------
    template<
        typename TAcc,
        typename TData,
        typename TExtent>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        TData const * const
#ifndef NDEBUG
        data
#endif
        ,
        TExtent const & extents,
        size_t const
#ifndef NDEBUG
        pitch
#endif
        ) const
    -> void
    {
        auto const globalThreadIdx = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent = alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearizedGlobalThreadIdx = alpaka::idx::mapIdx<1u>(
            globalThreadIdx,
            globalThreadExtent);

        for(size_t i(linearizedGlobalThreadIdx[0]); i < extents.prod(); i += globalThreadExtent.prod())
        {
            ALPAKA_ASSERT(data[linIdxToPitchedIdx<2>(i,pitch)] == i);
        }
    }
};

//#############################################################################
//! Fills values of buffer with increasing elements starting from 0
struct FillBufferKernel
{
    template<
        typename TAcc,
        typename TData,
        typename TExtent>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        TData * const data,
        TExtent const & extents) const
    -> void
    {
        auto const globalThreadIdx = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent = alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearizedGlobalThreadIdx = alpaka::idx::mapIdx<1u>(
            globalThreadIdx,
            globalThreadExtent);

        for(size_t i(linearizedGlobalThreadIdx[0]); i < extents.prod(); i += globalThreadExtent.prod())
        {
            data[i] = static_cast<TData>(i);
        }
    }
};

auto main()
-> int
{
// This example is hard-coded to use the sequential executor.
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)

    // Define the index domain
    using Dim = alpaka::dim::DimInt<3u>;
    using Idx = std::size_t;

    // Define the accelerator
    using Acc = alpaka::acc::AccCpuSerial<Dim, Idx>;
    using DevQueue = alpaka::queue::QueueCpuSync;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;

    using Host = alpaka::acc::AccCpuSerial<Dim, Idx>;
    using HostQueue = alpaka::queue::QueueCpuSync;
    using DevHost = alpaka::dev::Dev<Host>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;

    // Select devices
    DevAcc const devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));
    DevHost const devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));

    // Create queues
    DevQueue devQueue(devAcc);
    HostQueue hostQueue(devHost);

    // Define the work division
    using Vec = alpaka::vec::Vec<Dim, Idx>;
    Vec const elementsPerThread(Vec::all(static_cast<Idx>(1)));
    Vec const threadsPerBlock(Vec::all(static_cast<Idx>(1)));

    Vec const blocksPerGrid(
        static_cast<Idx>(4),
        static_cast<Idx>(8),
        static_cast<Idx>(16));

    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Idx>;
    WorkDiv const workdiv(
        blocksPerGrid,
        threadsPerBlock,
        elementsPerThread);


    // Create host and device buffers
    //
    // A buffer is an n-dimensional structure with a
    // particular data type and size which corresponds
    // to memory on the desired device. Buffers can be
    // allocated on the device or can be obtained from
    // already existing allocations e.g. std::array,
    // std::vector or a simple call to new.
    using Data = std::uint32_t;
    constexpr Idx nElementsPerDim = 2;

    const Vec extents(Vec::all(static_cast<Idx>(nElementsPerDim)));

    // Allocate host memory buffers
    //
    // The `alloc` method returns a reference counted buffer handle.
    // When the last such handle is destroyed, the memory is freed automatically.
    using BufHost = alpaka::mem::buf::Buf<DevHost, Data, Dim, Idx>;
    BufHost hostBuffer(alpaka::mem::buf::alloc<Data, Idx>(devHost, extents));
    // You can also use already allocated memory and wrap it within a view (irrespective of the device type).
    // The view does not own the underlying memory. So you have to make sure that
    // the view does not outlive its underlying memory.
    std::array<Data, nElementsPerDim * nElementsPerDim * nElementsPerDim> plainBuffer;
    using ViewHost = alpaka::mem::view::ViewPlainPtr<DevHost, Data, Dim, Idx>;
    ViewHost hostViewPlainPtr(plainBuffer.data(), devHost, extents);

    // Allocate accelerator memory buffers
    //
    // The interface to allocate a buffer is the same on the host and on the device.
    using BufAcc = alpaka::mem::buf::Buf<DevAcc, Data, Dim, Idx>;
    BufAcc deviceBuffer1(alpaka::mem::buf::alloc<Data, Idx>(devAcc, extents));
    BufAcc deviceBuffer2(alpaka::mem::buf::alloc<Data, Idx>(devAcc, extents));


    // Init host buffer
    //
    // You can not access the inner
    // elements of a buffer directly, but
    // you can get the pointer to the memory
    // (getPtrNative).
    Data * const pHostBuffer = alpaka::mem::view::getPtrNative(hostBuffer);

    // This pointer can be used to directly write
    // some values into the buffer memory.
    // Mind, that only a host can write on host memory.
    // The same holds true for device memory.
    for(Idx i(0); i < extents.prod(); ++i)
    {
        pHostBuffer[i] = static_cast<Data>(i);
    }

    // Memory views and buffers can also be initialized by executing a kernel.
    // To pass a buffer into a kernel, you can pass the
    // native pointer into the kernel invocation.
    Data * const pHostViewPlainPtr = alpaka::mem::view::getPtrNative(hostViewPlainPtr);

    FillBufferKernel fillBufferKernel;

    alpaka::kernel::exec<Host>(
        hostQueue,
        workdiv,
        fillBufferKernel,
        pHostViewPlainPtr, // 1st kernel argument
        extents);          // 2nd kernel argument


    // Copy host to device Buffer
    //
    // A copy operation of one buffer into
    // another buffer is enqueued into a queue
    // like it is done for kernel execution.
    // As always within alpaka, you will get a compile
    // time error if the desired copy coperation
    // (e.g. between various accelerator devices) is
    // not currently supported.
    // In this example both host buffers are copied
    // into device buffers.
    alpaka::mem::view::copy(devQueue, deviceBuffer1, hostViewPlainPtr, extents);
    alpaka::mem::view::copy(devQueue, deviceBuffer2, hostBuffer, extents);

    Idx const deviceBuffer1Pitch(alpaka::mem::view::getPitchBytes<2u>(deviceBuffer1) / sizeof(Data));
    Idx const deviceBuffer2Pitch(alpaka::mem::view::getPitchBytes<2u>(deviceBuffer2) / sizeof(Data));
    Idx const hostBuffer1Pitch(alpaka::mem::view::getPitchBytes<2u>(hostBuffer) / sizeof(Data));
    Idx const hostViewPlainPtrPitch(alpaka::mem::view::getPitchBytes<2u>(hostViewPlainPtr) / sizeof(Data));

    // Test device Buffer
    //
    // This kernel tests if the copy operations
    // were successful. In the case something
    // went wrong an assert will fail.
    Data const * const pDeviceBuffer1 = alpaka::mem::view::getPtrNative(deviceBuffer1);
    Data const * const pDeviceBuffer2 = alpaka::mem::view::getPtrNative(deviceBuffer2);

    TestBufferKernel testBufferKernel;
    alpaka::kernel::exec<Acc>(
        devQueue,
        workdiv,
        testBufferKernel,
        pDeviceBuffer1,                                 // 1st kernel argument
        extents,                                        // 2nd kernel argument
        deviceBuffer1Pitch);                            // 3rd kernel argument

    alpaka::kernel::exec<Acc>(
        devQueue,
        workdiv,
        testBufferKernel,
        pDeviceBuffer2,                                 // 1st kernel argument
        extents,                                        // 2nd kernel argument
        deviceBuffer2Pitch);                            // 3rd kernel argument


    // Print device Buffer
    //
    // Because we really like to flood our
    // terminal with numbers, the following
    // kernel prints all numbers of the
    // device buffer to stdout on the terminal.
    // Since this possibly is a parallel operation,
    // the output can appear in any order or even
    // completely distorted.

    PrintBufferKernel printBufferKernel;
    alpaka::kernel::exec<Acc>(
        devQueue,
        workdiv,
        printBufferKernel,
        pDeviceBuffer1,                                 // 1st kernel argument
        extents,                                        // 2nd kernel argument
        deviceBuffer1Pitch);                            // 3rd kernel argument
    std::cout << std::endl;

    alpaka::kernel::exec<Acc>(
        devQueue,
        workdiv,
        printBufferKernel,
        pDeviceBuffer2,                                 // 1st kernel argument
        extents,                                        // 2nd kernel argument
        deviceBuffer2Pitch);                            // 3rd kernel argument
    std::cout << std::endl;

    alpaka::kernel::exec<Host>(
        hostQueue,
        workdiv,
        printBufferKernel,
        pHostBuffer,                                    // 1st kernel argument
        extents,                                        // 2nd kernel argument
        hostBuffer1Pitch);                              // 3rd kernel argument
    std::cout << std::endl;

    alpaka::kernel::exec<Host>(
        hostQueue,
        workdiv,
        printBufferKernel,
        pHostViewPlainPtr,                              // 1st kernel argument
        extents,                                        // 2nd kernel argument
        hostViewPlainPtrPitch);                         // 3rd kernel argument
    std::cout << std::endl;

    return EXIT_SUCCESS;

#else
    return EXIT_SUCCESS;
#endif
}
