/*
  mallocMC: Memory Allocator for Many Core Architectures.
  https://www.hzdr.de/crp

  Copyright 2014 - 2024 Institute of Radiation Physics,
                 Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Carlchristian Eckert - c.eckert ( at ) hzdr.de
              Julian Lenz - j.lenz ( at ) hzdr.de

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/

#include "mallocMC/creationPolicies/OldMalloc.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <mallocMC/mallocMC.hpp>

#include <algorithm>
#include <iostream>

using mallocMC::CreationPolicies::FlatterScatter;
using mallocMC::CreationPolicies::OldMalloc;
using mallocMC::CreationPolicies::Scatter;

using Dim = alpaka::DimInt<1>;
using Idx = std::size_t;

// Define the device accelerator
using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;

constexpr uint32_t const blocksize = 2U * 1024U * 1024U;
constexpr uint32_t const pagesize = 4U * 1024U;
constexpr uint32_t const wasteFactor = 1U;

// This happens to also work for the original Scatter algorithm, so we only define one.
struct FlatterScatterHeapConfig : FlatterScatter<>::Properties::HeapConfig
{
    static constexpr auto accessblocksize = blocksize;
    static constexpr auto pagesize = ::pagesize;
    static constexpr auto heapsize = 2U * 1024U * 1024U * 1024U;
    // Only used by original Scatter (but it doesn't hurt FlatterScatter to keep):
    static constexpr auto regionsize = 16;
    static constexpr auto wastefactor = wasteFactor;
};

struct AlignmentConfig
{
    static constexpr auto dataAlignment = 16;
};

ALPAKA_STATIC_ACC_MEM_GLOBAL int* arA = nullptr;

template<typename T_Allocator>
struct ExampleKernel
{
    ALPAKA_FN_ACC void operator()(Acc const& acc, T_Allocator::AllocatorHandle allocHandle) const
    {
        auto const id = static_cast<uint32_t>(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0]);
        if(id == 0)
        {
            arA<Acc> = static_cast<int*>(allocHandle.malloc(acc, sizeof(int) * 32U));
        }
        // wait the the malloc from thread zero is not changing the result for some threads
        alpaka::syncBlockThreads(acc);
        auto const slots = allocHandle.getAvailableSlots(acc, 1);
        if(arA<Acc> != nullptr)
        {
            arA<Acc>[id] = id;
            printf("id: %u array: %d slots %u\n", id, arA<Acc>[id], slots);
        }
        else
            printf("error: device size allocation failed");

        // wait that all thread read from `arA<Acc>`
        alpaka::syncBlockThreads(acc);
        if(id == 0)
        {
            allocHandle.free(acc, arA<Acc>);
        }
    }
};

template<typename T_CreationPolicy>
auto example03() -> int
{
    using Allocator = mallocMC::Allocator<
        Acc,
        T_CreationPolicy,
        mallocMC::DistributionPolicies::Noop,
        mallocMC::OOMPolicies::ReturnNull,
        mallocMC::ReservePoolPolicies::AlpakaBuf<Acc>,
        mallocMC::AlignmentPolicies::Shrink<AlignmentConfig>>;

    auto const platform = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platform, 0);
    auto queue = alpaka::Queue<Acc, alpaka::Blocking>{dev};
    auto const devProps = alpaka::getAccDevProps<Acc>(dev);
    unsigned const block = std::min(static_cast<size_t>(32U), static_cast<size_t>(devProps.m_blockThreadCountMax));

    Allocator scatterAlloc(dev, queue, 2U * 1024U * 1024U * 1024U); // 2GB for device-side malloc

    auto const workDiv = alpaka::WorkDivMembers<Dim, Idx>{Idx{1}, Idx{block}, Idx{1}};
    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<Acc>(workDiv, ExampleKernel<Allocator>{}, scatterAlloc.getAllocatorHandle()));

    std::cout << "Slots from Host: " << scatterAlloc.getAvailableSlots(dev, queue, 1) << '\n';

    return 0;
}

auto main(int /*argc*/, char* /*argv*/[]) -> int
{
    example03<FlatterScatter<FlatterScatterHeapConfig>>();
    example03<Scatter<FlatterScatterHeapConfig>>();
    example03<OldMalloc>();
    return 0;
}
