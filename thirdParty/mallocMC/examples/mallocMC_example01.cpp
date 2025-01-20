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

#include "mallocMC/creationPolicies/FlatterScatter.hpp"
#include "mallocMC/creationPolicies/OldMalloc.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <mallocMC/mallocMC.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>

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

struct XMallocConfig
{
    static constexpr auto pagesize = FlatterScatterHeapConfig::pagesize;
};

struct ShrinkConfig
{
    static constexpr auto dataAlignment = 16;
};

ALPAKA_STATIC_ACC_MEM_GLOBAL int** arA;
ALPAKA_STATIC_ACC_MEM_GLOBAL int** arB;
ALPAKA_STATIC_ACC_MEM_GLOBAL int** arC;

template<typename T_CreationPolicy>
auto example01() -> int
{
    using Allocator = mallocMC::Allocator<
        Acc,
        T_CreationPolicy,
        mallocMC::DistributionPolicies::Noop,
        mallocMC::OOMPolicies::ReturnNull,
        mallocMC::ReservePoolPolicies::AlpakaBuf<Acc>,
        mallocMC::AlignmentPolicies::Shrink<ShrinkConfig>>;

    constexpr auto length = 100;

    auto const platform = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platform, 0);
    auto queue = alpaka::Queue<Acc, alpaka::Blocking>{dev};

    auto const devProps = alpaka::getAccDevProps<Acc>(dev);
    unsigned const block = std::min(static_cast<size_t>(32U), static_cast<size_t>(devProps.m_blockThreadCountMax));

    // round up
    auto grid = (length + block - 1U) / block;
    assert(length <= block * grid); // necessary for used algorithm

    // init the heap
    std::cerr << "initHeap...";
    auto const heapSize = 2U * 1024U * 1024U * 1024U;
    Allocator scatterAlloc(dev, queue, heapSize); // 1GB for device-side malloc
    std::cerr << "done\n";
    std::cout << Allocator::info("\n") << '\n';

    // create arrays of arrays on the device
    {
        auto createArrayPointers
            = [] ALPAKA_FN_ACC(Acc const& acc, int x, int y, Allocator::AllocatorHandle allocHandle)
        {
            arA<Acc> = static_cast<int**>(allocHandle.malloc(acc, sizeof(int*) * x * y));
            arB<Acc> = static_cast<int**>(allocHandle.malloc(acc, sizeof(int*) * x * y));
            arC<Acc> = static_cast<int**>(allocHandle.malloc(acc, sizeof(int*) * x * y));
        };
        auto const workDiv = alpaka::WorkDivMembers<Dim, Idx>{Idx{1}, Idx{1}, Idx{1}};
        alpaka::enqueue(
            queue,
            alpaka::createTaskKernel<Acc>(
                workDiv,
                createArrayPointers,
                grid,
                block,
                scatterAlloc.getAllocatorHandle()));
    }

    // fill 2 of them all with ascending values
    {
        auto fillArrays = [] ALPAKA_FN_ACC(Acc const& acc, int localLength, Allocator::AllocatorHandle allocHandle)
        {
            auto const id = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

            arA<Acc>[id] = static_cast<int*>(allocHandle.malloc(acc, localLength * sizeof(int)));
            arB<Acc>[id] = static_cast<int*>(allocHandle.malloc(acc, localLength * sizeof(int)));
            arC<Acc>[id] = static_cast<int*>(allocHandle.malloc(acc, localLength * sizeof(int)));

            for(int i = 0; i < localLength; ++i)
            {
                arA<Acc>[id][i] = static_cast<int>(id * localLength + i);
                arB<Acc>[id][i] = static_cast<int>(id * localLength + i);
            }
        };
        auto const workDiv = alpaka::WorkDivMembers<Dim, Idx>{Idx{grid}, Idx{block}, Idx{1}};
        alpaka::enqueue(
            queue,
            alpaka::createTaskKernel<Acc>(workDiv, fillArrays, length, scatterAlloc.getAllocatorHandle()));
    }

    // add the 2 arrays (vector addition within each thread)
    // and do a thread-wise reduce to sums
    {
        auto sumsBufferAcc = alpaka::allocBuf<int, Idx>(dev, Idx{block * grid});

        auto addArrays = [] ALPAKA_FN_ACC(Acc const& acc, int localLength, int* sums)
        {
            auto const id = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

            sums[id] = 0;
            for(int i = 0; i < localLength; ++i)
            {
                arC<Acc>[id][i] = arA<Acc>[id][i] + arB<Acc>[id][i];
                sums[id] += arC<Acc>[id][i];
            }
        };
        auto const workDiv = alpaka::WorkDivMembers<Dim, Idx>{Idx{grid}, Idx{block}, Idx{1}};
        alpaka::enqueue(
            queue,
            alpaka::createTaskKernel<Acc>(workDiv, addArrays, length, alpaka::getPtrNative(sumsBufferAcc)));

        auto const platformCPU = alpaka::Platform<alpaka::DevCpu>{};
        auto const hostDev = alpaka::getDevByIdx(platformCPU, 0);

        auto sumsBufferHost = alpaka::allocBuf<int, Idx>(hostDev, Idx{block * grid});
        alpaka::memcpy(queue, sumsBufferHost, sumsBufferAcc, Idx{block * grid});
        alpaka::wait(queue);

        auto const* sumsPtr = alpaka::getPtrNative(sumsBufferHost);
        auto const sum = std::accumulate(sumsPtr, sumsPtr + block * grid, size_t{0});
        std::cout << "The sum of the arrays on GPU is " << sum << '\n';
    }

    auto const n = static_cast<size_t>(block * grid * length);
    auto const gaussian = n * (n - 1);
    std::cout << "The gaussian sum as comparison: " << gaussian << '\n';

    /*constexpr*/ if(mallocMC::Traits<Allocator>::providesAvailableSlots)
    {
        std::cout << "there are ";
        std::cout << scatterAlloc.getAvailableSlots(dev, queue, 1024U * 1024U);
        std::cout << " Slots of size 1MB available\n";
    }

    {
        auto freeArrays = [] ALPAKA_FN_ACC(Acc const& acc, Allocator::AllocatorHandle allocHandle)
        {
            auto const id = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
            allocHandle.free(acc, arA<Acc>[id]);
            allocHandle.free(acc, arB<Acc>[id]);
            allocHandle.free(acc, arC<Acc>[id]);
        };
        auto const workDiv = alpaka::WorkDivMembers<Dim, Idx>{Idx{grid}, Idx{block}, Idx{1}};
        alpaka::enqueue(queue, alpaka::createTaskKernel<Acc>(workDiv, freeArrays, scatterAlloc.getAllocatorHandle()));
    }

    {
        auto freeArrayPointers = [] ALPAKA_FN_ACC(Acc const& acc, Allocator::AllocatorHandle allocHandle)
        {
            allocHandle.free(acc, arA<Acc>);
            allocHandle.free(acc, arB<Acc>);
            allocHandle.free(acc, arC<Acc>);
        };
        auto const workDiv = alpaka::WorkDivMembers<Dim, Idx>{Idx{1}, Idx{1}, Idx{1}};
        alpaka::enqueue(
            queue,
            alpaka::createTaskKernel<Acc>(workDiv, freeArrayPointers, scatterAlloc.getAllocatorHandle()));
    }

    return 0;
}

auto main(int /*argc*/, char* /*argv*/[]) -> int
{
    example01<FlatterScatter<FlatterScatterHeapConfig>>();
    example01<Scatter<FlatterScatterHeapConfig>>();
    example01<OldMalloc>();
    return 0;
}
