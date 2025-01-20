/* Copyright 2024 Alexander Matthes, Benjamin Worpitz, Erik Zenker, Matthias Werner, Bernhard Manfred Gruber,
 *                Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>

#include <cstdint>
#include <iostream>

//! Prints all elements of the buffer.
struct PrintBufferKernel
{
    template<typename TAcc, typename MdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, MdSpan data) const -> void
    {
        // Use three nested loops along the dimensions 0, 1 and 2
        for(size_t z : alpaka::uniformElementsAlong<0>(acc, data.extent(0)))
            for(size_t y : alpaka::uniformElementsAlong<1>(acc, data.extent(1)))
                for(size_t x : alpaka::uniformElementsAlong<2>(acc, data.extent(2)))
                    // %zu prints garbage in some cases, while %lu seems to be working correctly
                    printf("%lu,%lu,%lu: %u\t", z, y, x, static_cast<uint32_t>(data(z, y, x)));
    }
};

//! Tests if the value of the buffer on index i is equal to i.
struct TestBufferKernel
{
    template<typename TAcc, typename MdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, MdSpan data) const -> void
    {
        // Use three nested loops along the dimensions z, y and x
        for(size_t z : alpaka::uniformElementsAlongZ(acc, data.extent(0)))
            for(size_t y : alpaka::uniformElementsAlongY(acc, data.extent(1)))
                for(size_t x : alpaka::uniformElementsAlongX(acc, data.extent(2)))
                    ALPAKA_ASSERT_ACC(
                        data(z, y, x)
                        == alpaka::mapIdx<1u>(
                            alpaka::Vec{z, y, x},
                            alpaka::Vec{data.extent(0), data.extent(1), data.extent(2)})[0]);
    }
};

//! Fills values of buffer with increasing elements starting from 0
struct FillBufferKernel
{
    template<typename TAcc, typename MdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, MdSpan data) const -> void
    {
        // Use a single 3-dimensional loop
        for(auto idx : alpaka::uniformElementsND(acc, alpaka::Vec{data.extent(0), data.extent(1), data.extent(2)}))
            data(idx.z(), idx.y(), idx.x()) // equivalent to data(idx[0], idx[1], idx[2])
                = alpaka::mapIdx<1u>(idx, alpaka::Vec{data.extent(0), data.extent(1), data.extent(2)})[0];
    }
};

// In standard projects, you typically do not execute the code with any available accelerator.
// Instead, a single accelerator is selected once from the active accelerators and the kernels are executed with the
// selected accelerator only. If you use the example as the starting point for your project, you can rename the
// example() function to main() and move the accelerator tag to the function body.
template<typename TAccTag>
auto example(TAccTag const&) -> int
{
    // Define the index domain
    using Dim = alpaka::DimInt<3u>;
    using Idx = std::size_t;

    // Define the device accelerator
    using Acc = alpaka::TagToAcc<TAccTag, Dim, Idx>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;
    // Defines the synchronization behavior of the device queue
    //
    // choose between Blocking and NonBlocking
    using AccQueueProperty = alpaka::Blocking;
    using DevQueue = alpaka::Queue<Acc, AccQueueProperty>;

    // Define the host accelerator
    using Host = alpaka::AccCpuSerial<Dim, Idx>;
    // Defines the synchronization behavior of the host queue
    //
    // choose between Blocking and NonBlocking
    using HostQueueProperty = alpaka::Blocking;
    using HostQueue = alpaka::Queue<Host, HostQueueProperty>;

    // Select devices
    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);
    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);

    // Create queues
    DevQueue devQueue(devAcc);
    HostQueue hostQueue(devHost);

    // Define the work division for kernels to be run on devAcc and devHost
    using Vec = alpaka::Vec<Dim, Idx>;
    Vec const elementsPerThread(Vec::all(static_cast<Idx>(3)));
    Vec const elementsPerGrid(Vec::all(static_cast<Idx>(10)));

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

    Vec const extents = Vec::all(nElementsPerDim);

    // Allocate host memory buffers
    //
    // The `alloc` method returns a reference counted buffer handle.
    // When the last such handle is destroyed, the memory is freed automatically.
    using BufHost = alpaka::Buf<Host, Data, Dim, Idx>;
    BufHost hostBuffer = alpaka::allocBuf<Data, Idx>(devHost, extents);
    // You can also use already allocated memory and wrap it within a view (irrespective of the device type).
    // The view does not own the underlying memory. So you have to make sure that
    // the view does not outlive its underlying memory.
    std::array<Data, nElementsPerDim * nElementsPerDim * nElementsPerDim> plainBuffer;
    auto hostViewPlainPtr = alpaka::createView(devHost, plainBuffer.data(), extents);

    // Allocate accelerator memory buffers
    //
    // The interface to allocate a buffer is the same on the host and on the device.
    using BufAcc = alpaka::Buf<Acc, Data, Dim, Idx>;
    BufAcc deviceBuffer1 = alpaka::allocBuf<Data, Idx>(devAcc, extents);
    BufAcc deviceBuffer2 = alpaka::allocBuf<Data, Idx>(devAcc, extents);


    // Init host buffer
    //
    // You can not access the inner
    // elements of a buffer directly, but
    // you can get the pointer to the memory
    // (getPtrNative).
    auto hostBufferMdSpan = alpaka::experimental::getMdSpan(hostBuffer);

    // This pointer can be used to directly write
    // some values into the buffer memory.
    // Mind, that only a host can write on host memory.
    // The same holds true for device memory.
    for(Idx z = 0; z < extents[0]; ++z)
        for(Idx y = 0; y < extents[1]; ++y)
            for(Idx x = 0; x < extents[2]; ++x)
                hostBufferMdSpan(z, y, x) = static_cast<Data>(z * extents[1] * extents[2] + y * extents[2] + x);

    // Memory views and buffers can also be initialized by executing a kernel.
    // To pass a buffer into a kernel, you can pass the
    // native pointer into the kernel invocation.
    auto hostViewPlainPtrMdSpan = alpaka::experimental::getMdSpan(hostViewPlainPtr);

    FillBufferKernel fillBufferKernel;

    alpaka::KernelCfg<Host> const hostKernelCfg = {elementsPerGrid, elementsPerThread};
    auto const hostWorkDiv = alpaka::getValidWorkDiv(hostKernelCfg, devHost, fillBufferKernel, hostViewPlainPtrMdSpan);

    alpaka::exec<Host>(hostQueue, hostWorkDiv, fillBufferKernel,
                       hostViewPlainPtrMdSpan); // 1st kernel argument

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
    alpaka::memcpy(devQueue, deviceBuffer1, hostViewPlainPtr);
    alpaka::memcpy(devQueue, deviceBuffer2, hostBuffer);

    // Depending on the accelerator, the allocation function may introduce
    // padding between rows/planes of multidimensional memory allocations.
    // Therefore the pitch (distance between consecutive rows/planes) may be
    // greater than the space required for the data.
    Idx const deviceBuffer1Pitch(alpaka::getPitchesInBytes(deviceBuffer1)[1] / sizeof(Data));
    Idx const deviceBuffer2Pitch(alpaka::getPitchesInBytes(deviceBuffer2)[1] / sizeof(Data));
    Idx const hostBuffer1Pitch(alpaka::getPitchesInBytes(hostBuffer)[1] / sizeof(Data));
    Idx const hostViewPlainPtrPitch(alpaka::getPitchesInBytes(hostViewPlainPtr)[1] / sizeof(Data));

    // Test device Buffer
    //
    // This kernel tests if the copy operations
    // were successful. In the case something
    // went wrong an assert will fail.
    auto deviceBufferMdSpan1 = alpaka::experimental::getMdSpan(deviceBuffer1);
    auto deviceBufferMdSpan2 = alpaka::experimental::getMdSpan(deviceBuffer2);

    TestBufferKernel testBufferKernel;

    // Let alpaka calculate good block and grid sizes given our full problem extent
    alpaka::KernelCfg<Acc> const devKernelCfg = {elementsPerGrid, elementsPerThread};
    auto const devWorkDiv = alpaka::getValidWorkDiv(devKernelCfg, devAcc, testBufferKernel, deviceBufferMdSpan1);

    alpaka::exec<Acc>(devQueue, devWorkDiv, testBufferKernel, deviceBufferMdSpan1);
    alpaka::exec<Acc>(devQueue, devWorkDiv, testBufferKernel, deviceBufferMdSpan2);

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

    // Let alpaka calculate good block and grid sizes given our full problem extent
    auto const hostPrintWorkDiv
        = alpaka::getValidWorkDiv(hostKernelCfg, devHost, printBufferKernel, hostViewPlainPtrMdSpan);
    auto const devPrintWorkDiv = alpaka::getValidWorkDiv(devKernelCfg, devAcc, printBufferKernel, deviceBufferMdSpan1);

    alpaka::exec<Acc>(devQueue, devPrintWorkDiv, printBufferKernel, deviceBufferMdSpan1);
    alpaka::wait(devQueue);
    std::cout << std::endl;

    alpaka::exec<Acc>(devQueue, devPrintWorkDiv, printBufferKernel, deviceBufferMdSpan2);
    alpaka::wait(devQueue);
    std::cout << std::endl;

    alpaka::exec<Host>(hostQueue, hostPrintWorkDiv, printBufferKernel, hostBufferMdSpan);
    alpaka::wait(hostQueue);
    std::cout << std::endl;

    alpaka::exec<Host>(hostQueue, hostPrintWorkDiv, printBufferKernel, hostViewPlainPtrMdSpan);
    alpaka::wait(hostQueue);
    std::cout << std::endl;

    return EXIT_SUCCESS;
}

auto main() -> int
{
    // Execute the example once for each enabled accelerator.
    // If you would like to execute it for a single accelerator only you can use the following code.
    //  \code{.cpp}
    //  auto tag = TagCpuSerial;
    //  return example(tag);
    //  \endcode
    //
    // valid tags:
    //   TagCpuSerial, TagGpuHipRt, TagGpuCudaRt, TagCpuOmp2Blocks, TagCpuTbbBlocks,
    //   TagCpuOmp2Threads, TagCpuSycl, TagCpuTbbBlocks, TagCpuThreads,
    //   TagFpgaSyclIntel, TagGenericSycl, TagGpuSyclIntel
    return alpaka::executeForEachAccTag([=](auto const& tag) { return example(tag); });
}
