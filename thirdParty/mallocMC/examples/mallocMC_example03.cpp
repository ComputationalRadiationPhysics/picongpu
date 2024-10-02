/*
  mallocMC: Memory Allocator for Many Core Architectures.
  https://www.hzdr.de/crp

  Copyright 2014 Institute of Radiation Physics,
                 Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Carlchristian Eckert - c.eckert ( at ) hzdr.de

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

#include <algorithm>
#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <cassert>
#include <iostream>
#include <mallocMC/mallocMC.hpp>
#include <numeric>
#include <vector>

using Dim = alpaka::DimInt<1>;
using Idx = std::size_t;

// Define the device accelerator
using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;

struct ScatterConfig
{
    static constexpr auto pagesize = 4096;
    static constexpr auto accessblocksize = 512u * 1024u * 1024u;
    static constexpr auto regionsize = 16;
    static constexpr auto wastefactor = 2;
    static constexpr auto resetfreedpages = false;
};

struct ScatterHashParams
{
    static constexpr auto hashingK = 38183;
    static constexpr auto hashingDistMP = 17497;
    static constexpr auto hashingDistWP = 1;
    static constexpr auto hashingDistWPRel = 1;
};

struct AlignmentConfig
{
    static constexpr auto dataAlignment = 16;
};

using ScatterAllocator = mallocMC::Allocator<
    Acc,
    mallocMC::CreationPolicies::Scatter<ScatterConfig, ScatterHashParams>,
    mallocMC::DistributionPolicies::Noop,
    mallocMC::OOMPolicies::ReturnNull,
    mallocMC::ReservePoolPolicies::AlpakaBuf<Acc>,
    mallocMC::AlignmentPolicies::Shrink<AlignmentConfig>>;

ALPAKA_STATIC_ACC_MEM_GLOBAL int* arA = nullptr;

struct ExampleKernel
{
    ALPAKA_FN_ACC void operator()(const Acc& acc, ScatterAllocator::AllocatorHandle allocHandle) const
    {
        const auto id = static_cast<uint32_t>(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0]);
        if(id == 0)
            arA = (int*) allocHandle.malloc(acc, sizeof(int) * 32);
        // wait the the malloc from thread zero is not changing the result for some threads
        alpaka::syncBlockThreads(acc);
        const auto slots = allocHandle.getAvailableSlots(acc, 1);
        if(arA != nullptr)
        {
            arA[id] = id;
            printf("id: %u array: %d slots %u\n", id, arA[id], slots);
        }
        else
            printf("error: device size allocation failed");

        // wait that all thread read from `arA`
        alpaka::syncBlockThreads(acc);
        if(id == 0)
            allocHandle.free(acc, arA);
    }
};

auto main() -> int
{
    auto const platform = alpaka::Platform<Acc>{};
    const auto dev = alpaka::getDevByIdx(platform, 0);
    auto queue = alpaka::Queue<Acc, alpaka::Blocking>{dev};
    auto const devProps = alpaka::getAccDevProps<Acc>(dev);
    unsigned const block = std::min(static_cast<size_t>(32u), static_cast<size_t>(devProps.m_blockThreadCountMax));

    ScatterAllocator scatterAlloc(dev, queue, 1U * 1024U * 1024U * 1024U); // 1GB for device-side malloc

    const auto workDiv = alpaka::WorkDivMembers<Dim, Idx>{Idx{1}, Idx{block}, Idx{1}};
    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc>(workDiv, ExampleKernel{}, scatterAlloc.getAllocatorHandle()));

    std::cout << "Slots from Host: " << scatterAlloc.getAvailableSlots(dev, queue, 1) << '\n';

    return 0;
}
