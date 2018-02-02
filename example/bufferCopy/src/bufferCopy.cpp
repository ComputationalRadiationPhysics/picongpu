/**
 * \file
 * Copyright 2014-2017 Erik Zenker, Benjamin Worpitz
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
#include <cassert>

#define DIMENSION 3

//#############################################################################

template <size_t width>
ALPAKA_FN_ACC size_t linIdxToPitchedIdx(size_t const globalIdx, size_t const pitch)
{
    const size_t idx_x = globalIdx % width;
    const size_t idx_y = globalIdx / width;
    return idx_x + idx_y * pitch;
}

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
            assert(data[linIdxToPitchedIdx<2>(i,pitch)] == i);
        }
    }
};

//#############################################################################
//! Inits all values of a buffer with initValue.
struct InitBufferKernel
{
    //-----------------------------------------------------------------------------
    template<
        typename TAcc,
        typename TData,
        typename TExtent>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        TData * const data,
        TExtent const & extents,
        TData const & initValue) const
    -> void
    {
        auto const globalThreadIdx = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent = alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearizedGlobalThreadIdx = alpaka::idx::mapIdx<1u>(
            globalThreadIdx,
            globalThreadExtent);

        for(size_t i(linearizedGlobalThreadIdx[0]); i < extents.prod(); i += globalThreadExtent.prod())
        {
            data[i] = initValue;
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
    // Configure types
    using Dim = alpaka::dim::DimInt<DIMENSION>;
    using Size = std::size_t;
    using Extents = Size;
    using Host = alpaka::acc::AccCpuSerial<Dim, Size>;
    using Acc = alpaka::acc::AccCpuSerial<Dim, Size>;
//    using Acc = alpaka::acc::AccCpuOmp2Blocks<Dim, Size>;
//    using Acc = alpaka::acc::AccGpuCudaRt<Dim, Size>;
    using DevHost = alpaka::dev::Dev<Host>;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Size>;
    using DevStream = alpaka::stream::StreamCpuSync;
//    using DevStream = alpaka::stream::StreamCudaRtSync;
    using HostStream = alpaka::stream::StreamCpuSync;

    // Get the first device
    DevAcc const devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));
    DevHost const devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));

    // Create sync stream
    DevStream devStream(devAcc);
    HostStream hostStream(devHost);


    // Init workdiv
    alpaka::vec::Vec<Dim, Size> const elementsPerThread(
#if DIMENSION > 2
        static_cast<Size>(1),
#endif
#if DIMENSION > 1
        static_cast<Size>(1),
#endif
        static_cast<Size>(1));

    alpaka::vec::Vec<Dim, Size> const threadsPerBlock(
#if DIMENSION > 2
        static_cast<Size>(1),
#endif
#if DIMENSION > 1
        static_cast<Size>(1),
#endif
        static_cast<Size>(1));

    alpaka::vec::Vec<Dim, Size> const blocksPerGrid(
#if DIMENSION > 2
        static_cast<Size>(4),
#endif
#if DIMENSION > 1
        static_cast<Size>(8),
#endif
        static_cast<Size>(16));

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
    constexpr Extents nElementsPerDim = 2;

    const alpaka::vec::Vec<Dim, Size> extents(
#if DIMENSION > 2
        static_cast<Size>(nElementsPerDim),
#endif
#if DIMENSION > 1
        static_cast<Size>(nElementsPerDim),
#endif
        static_cast<Size>(nElementsPerDim));

    std::array<Data, nElementsPerDim * nElementsPerDim * nElementsPerDim> plainBuffer;
    alpaka::mem::view::ViewPlainPtr<DevHost, Data, Dim, Size> hostBufferPlain(plainBuffer.data(), devHost, extents);
    alpaka::mem::buf::Buf<DevHost, Data, Dim, Size> hostBuffer(alpaka::mem::buf::alloc<Data, Size>(devHost, extents));
    alpaka::mem::buf::Buf<DevAcc, Data, Dim, Size> deviceBuffer1(alpaka::mem::buf::alloc<Data, Size>(devAcc, extents));
    alpaka::mem::buf::Buf<DevAcc, Data, Dim, Size> deviceBuffer2(alpaka::mem::buf::alloc<Data, Size>(devAcc, extents));


    // Init plain host buffer
    //
    // The buffer obtained from a plain pointer
    // of the host is initialized here with the
    // value zero. It is recommended to use a
    // kernel for such a task, since a kernel
    // can speed up the initialization process.
    // The buffer can be provided to a kernel
    // by passing the native pointer of the memory
    // within the buffer (getPtrNative).
    InitBufferKernel initBufferKernel;
    Data const initValue = 0u;

    auto const init(
        alpaka::exec::create<Host>(
            workdiv,
            initBufferKernel,
            alpaka::mem::view::getPtrNative(hostBuffer), // 1st kernel argument
            extents,                                     // 2nd kernel argument
            initValue));                                 // 3rd kernel argument

    alpaka::stream::enqueue(hostStream, init);


    // Write some data to the host buffer
    //
    // The buffer can not access the inner
    // elements by some access operator
    // directly, but it can return the
    // pointer to the memory within the
    // buffer (getPtrNative). This pointer
    // can be used to write some values into
    // the buffer memory. Mind, that only a host
    // can write on host memory. The same holds
    // for device memory.
    for(size_t i(0); i < extents.prod(); ++i)
    {
        alpaka::mem::view::getPtrNative(hostBuffer)[i] = static_cast<Data>(i);
    }



    // Fill plain host with increasing data
    //
    // A buffer can also be filled by a special
    // buffer fill kernel. This has the advantage,
    // that the kernel can be performed on any
    // buffer (device or host) and can be
    // run in parallel.
    FillBufferKernel fillBufferKernel;
    auto const fill(
        alpaka::exec::create<Host>(
            workdiv,
            fillBufferKernel,
            alpaka::mem::view::getPtrNative(hostBufferPlain), // 1st kernel argument
            extents));                                        // 2nd kernel argument

    alpaka::stream::enqueue(hostStream, fill);


    // Copy host to device Buffer
    //
    // A copy operation of one buffer into
    // another buffer is enqueued into a stream
    // like it is done for kernel execution, but
    // more automatically. Copy is only possible
    // from host to host, host to device and
    // device to host. Some devices also support
    // device to device copy, but this is not true
    // in general. However, currently all
    // (both CPU and GPU) devices support it.
    // As always within alpaka, you will get a compile
    // time error if the desired copy coperation
    // (e.g. between various accelerator devices) is
    // not currently supported.
    // In this example both host buffers are copied
    // into device buffers.
    alpaka::mem::view::copy(devStream, deviceBuffer1, hostBufferPlain, extents);
    alpaka::mem::view::copy(devStream, deviceBuffer2, hostBuffer, extents);

    auto devicePitch(alpaka::mem::view::getPitchBytes<DIMENSION-1>(deviceBuffer1) / sizeof(Data));
    auto hostPitch(alpaka::mem::view::getPitchBytes<DIMENSION-1>(hostBuffer) / sizeof(Data));

    // Test device Buffer
    //
    // This kernel tests if the copy operations
    // were successful. In the case something
    // went wrong an assert will fail.
    TestBufferKernel testBufferKernel;
    auto const test1(
        alpaka::exec::create<Acc>(
            workdiv,
            testBufferKernel,
            alpaka::mem::view::getPtrNative(deviceBuffer1), // 1st kernel argument
            extents,                                        // 2nd kernel argument
            devicePitch));                                  // 3rd kernel argument

    auto const test2(
        alpaka::exec::create<Acc>(
            workdiv,
            testBufferKernel,
            alpaka::mem::view::getPtrNative(deviceBuffer1), // 1st kernel argument
            extents,                                        // 2nd kernel argument
            devicePitch));                                  // 3rd kernel argument

    alpaka::stream::enqueue(devStream, test1);
    alpaka::stream::enqueue(devStream, test2);


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
    auto const printDeviceBuffer1(
        alpaka::exec::create<Acc>(
            workdiv,
            printBufferKernel,
            alpaka::mem::view::getPtrNative(deviceBuffer1),    // 1st kernel argument
            extents,                                           // 2nd kernel argument
            devicePitch));                                     // 3rd kernel argument
    auto const printDeviceBuffer2(
        alpaka::exec::create<Acc>(
            workdiv,
            printBufferKernel,
            alpaka::mem::view::getPtrNative(deviceBuffer2), // 1st kernel argument
            extents,                                        // 2nd kernel argument
            devicePitch));                                  // 3rd kernel argument

    auto const printHostBuffer(
        alpaka::exec::create<Host>(
            workdiv,
            printBufferKernel,
            alpaka::mem::view::getPtrNative(hostBuffer), // 1st kernel argument
            extents,                                     // 2nd kernel argument
            hostPitch));                                 // 3rd kernel argument

    auto const printHostBufferPlain(
        alpaka::exec::create<Host>(
            workdiv,
            printBufferKernel,
            alpaka::mem::view::getPtrNative(hostBufferPlain), // 1st kernel argument
            extents,                                          // 2nd kernel argument
            hostPitch));                                      // 3rd kernel argument

    alpaka::stream::enqueue(devStream, printDeviceBuffer1);
    std::cout << std::endl;
    alpaka::stream::enqueue(devStream, printDeviceBuffer2);
    std::cout << std::endl;
    alpaka::stream::enqueue(hostStream, printHostBuffer);
    std::cout << std::endl;
    alpaka::stream::enqueue(hostStream, printHostBufferPlain);
    std::cout << std::endl;


    // No copy failure, so lets return :)
    return EXIT_SUCCESS;
}
