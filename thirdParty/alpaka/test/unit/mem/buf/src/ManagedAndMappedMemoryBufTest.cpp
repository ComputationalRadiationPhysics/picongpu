/* Copyright 2025 Maria Michailidi
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/test/Extent.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/mem/view/ViewTest.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch_message.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <numeric>
#include <type_traits>

// Kernel that adds a value to the elements of a buffer
struct ValueAddKernel
{
    template<typename Acc, typename TElem, typename TIdx>
    ALPAKA_FN_ACC void operator()(Acc const& acc, TElem* data, int value, TIdx numElements) const
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

        if(idx < numElements)
        {
            data[idx] += value;
        }
    }
};

TEMPLATE_LIST_TEST_CASE("memBufManagedTest", "[memBuf]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dev = alpaka::Dev<Acc>;
    using Queue = alpaka::test::DefaultQueue<Dev>;
    using Elem = int;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);

    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platformAcc, 0);

    INFO("Test if unified memory works in: ");
    INFO(alpaka::getName(dev));

    Queue queue(dev);

    // Configure a 1D work division large enough to cover all elements
    Idx const numElements(123456);
    Idx const elementsPerThread(1u);

    alpaka::Vec<Dim, Idx> extent = alpaka::Vec<Dim, Idx>::ones();
    extent[0] = numElements;

    alpaka::Vec<Dim, Idx> elemsPerThreadVec = alpaka::Vec<Dim, Idx>::ones();
    elemsPerThreadVec[0] = elementsPerThread;

    // Allocate unified memory associated with the host device, accessible by both host and accelerator
    auto buf = alpaka::allocManagedBuf<Elem, Idx>(devHost, platformAcc, extent);

    constexpr Elem fillVal = 42;
    auto* hostPtr = alpaka::getPtrNative(buf);
    for(Idx i = 0; i < numElements; ++i)
    {
        hostPtr[i] = fillVal;
    }

    constexpr int value = 10;

    ValueAddKernel kernel;
    alpaka::KernelCfg<Acc> const kernelCfg = {extent, elemsPerThreadVec};

    // Calculate work division
    auto const workDiv = alpaka::getValidWorkDiv(kernelCfg, dev, kernel, buf.data(), value, numElements);

    std::cout << "Testing Kernel with scalar indices with a grid of "
              << alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(workDiv) << " blocks x "
              << alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(workDiv) << " threads x "
              << alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(workDiv) << " elements...\n";

    // Call of kernel to change the values of the buffer's elements
    auto const taskKernel = alpaka::createTaskKernel<Acc>(workDiv, kernel, buf.data(), value, numElements);

    alpaka::enqueue(queue, taskKernel);
    alpaka::wait(queue);

    // Verify that device writes are immediately visible on host without explicit copy
    bool passed = true;
    for(Idx i = 0; i < numElements; ++i)
    {
        if(hostPtr[i] != fillVal + value)
        {
            passed = false;
            break;
        }
    }
    CHECK(passed);
}

TEMPLATE_LIST_TEST_CASE("memBufMappedTest", "[memBuf]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dev = alpaka::Dev<Acc>;
    using Queue = alpaka::test::DefaultQueue<Dev>;
    using Elem = int;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);

    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platformAcc, 0);

    INFO("Test if mapped memory works in: ");
    INFO(alpaka::getName(dev));

    Queue queue(dev);

    // Configure a 1D work division large enough to cover all elements
    Idx const numElements(123456);
    Idx const elementsPerThread(1u);

    alpaka::Vec<Dim, Idx> extent = alpaka::Vec<Dim, Idx>::ones();
    extent[0] = numElements;

    alpaka::Vec<Dim, Idx> elemsPerThreadVec = alpaka::Vec<Dim, Idx>::ones();
    elemsPerThreadVec[0] = elementsPerThread;

    // Allocate pinned host memory mapped to the accelerator, accessible by both host and device
    auto buf = alpaka::allocMappedBuf<Elem, Idx>(devHost, platformAcc, extent);

    constexpr Elem fillVal = 42;
    auto* hostPtr = alpaka::getPtrNative(buf);
    for(Idx i = 0; i < numElements; ++i)
    {
        hostPtr[i] = fillVal;
    }

    constexpr int value = 10;

    ValueAddKernel kernel;
    alpaka::KernelCfg<Acc> const kernelCfg = {extent, elemsPerThreadVec};

    // Calculate work division
    auto const workDiv = alpaka::getValidWorkDiv(kernelCfg, dev, kernel, buf.data(), value, numElements);

    std::cout << "Testing Kernel with scalar indices with a grid of "
              << alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(workDiv) << " blocks x "
              << alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(workDiv) << " threads x "
              << alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(workDiv) << " elements...\n";

    // Call of kernel to change the values of the buffer's elements
    auto const taskKernel = alpaka::createTaskKernel<Acc>(workDiv, kernel, buf.data(), value, numElements);

    alpaka::enqueue(queue, taskKernel);
    alpaka::wait(queue);

    // Verify that device writes to the mapped host memory are visible on the host without an explicit copy
    bool passed = true;
    for(Idx i = 0; i < numElements; ++i)
    {
        if(hostPtr[i] != fillVal + value)
        {
            passed = false;
            break;
        }
    }
    CHECK(passed);
}
