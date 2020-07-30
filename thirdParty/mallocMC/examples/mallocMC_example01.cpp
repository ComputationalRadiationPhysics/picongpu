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

#include <alpaka/alpaka.hpp>
#include <iostream>
#include <mallocMC/mallocMC.hpp>
#include <numeric>

using Dim = alpaka::dim::DimInt<1>;
using Idx = std::size_t;
// using Acc = alpaka::acc::AccCpuThreads<Dim, Idx>;
// using Acc = alpaka::acc::AccCpuOmp2Threads<Dim, Idx>;
using Acc = alpaka::acc::AccGpuCudaRt<Dim, Idx>;

struct ScatterHeapConfig
{
    static constexpr auto pagesize = 4096;
    static constexpr auto accessblocks = 8;
    static constexpr auto regionsize = 16;
    static constexpr auto wastefactor = 2;
    static constexpr auto resetfreedpages = false;
};

struct ScatterHashConfig
{
    static constexpr auto hashingK = 38183;
    static constexpr auto hashingDistMP = 17497;
    static constexpr auto hashingDistWP = 1;
    static constexpr auto hashingDistWPRel = 1;
};

struct XMallocConfig
{
    static constexpr auto pagesize = ScatterHeapConfig::pagesize;
};

struct ShrinkConfig
{
    static constexpr auto dataAlignment = 16;
};

using ScatterAllocator = mallocMC::Allocator<
    Acc,
    mallocMC::CreationPolicies::Scatter<ScatterHeapConfig, ScatterHashConfig>,
    mallocMC::DistributionPolicies::XMallocSIMD<XMallocConfig>,
    mallocMC::OOMPolicies::ReturnNull,
    mallocMC::ReservePoolPolicies::AlpakaBuf<Acc>,
    mallocMC::AlignmentPolicies::Shrink<ShrinkConfig>>;

ALPAKA_STATIC_ACC_MEM_GLOBAL int ** arA;
ALPAKA_STATIC_ACC_MEM_GLOBAL int ** arB;
ALPAKA_STATIC_ACC_MEM_GLOBAL int ** arC;

auto main() -> int
{
    constexpr auto block = 32;
    constexpr auto grid = 32;
    constexpr auto length = 100;
    static_assert(length <= block * grid, ""); // necessary for used algorithm

    const auto dev = alpaka::pltf::getDevByIdx<Acc>(0);
    auto queue = alpaka::queue::Queue<Acc, alpaka::queue::Blocking>{dev};

    // init the heap
    std::cerr << "initHeap...";
    ScatterAllocator scatterAlloc(
        dev, queue, 1U * 1024U * 1024U * 1024U); // 1GB for device-side malloc
    std::cerr << "done\n";
    std::cout << ScatterAllocator::info("\n") << '\n';

    // create arrays of arrays on the device
    {
        auto createArrayPointers
            = [] ALPAKA_FN_ACC(
                  const Acc & acc,
                  int x,
                  int y,
                  ScatterAllocator::AllocatorHandle allocHandle) {
                  arA = (int **)allocHandle.malloc(acc, sizeof(int *) * x * y);
                  arB = (int **)allocHandle.malloc(acc, sizeof(int *) * x * y);
                  arC = (int **)allocHandle.malloc(acc, sizeof(int *) * x * y);
              };
        const auto workDiv
            = alpaka::workdiv::WorkDivMembers<Dim, Idx>{Idx{1}, Idx{1}, Idx{1}};
        alpaka::queue::enqueue(
            queue,
            alpaka::kernel::createTaskKernel<Acc>(
                workDiv,
                createArrayPointers,
                grid,
                block,
                scatterAlloc.getAllocatorHandle()));
    }

    // fill 2 of them all with ascending values
    {
        auto fillArrays = [] ALPAKA_FN_ACC(
                              const Acc & acc,
                              int length,
                              ScatterAllocator::AllocatorHandle allocHandle) {
            const auto id
                = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

            arA[id] = (int *)allocHandle.malloc(acc, length * sizeof(int));
            arB[id] = (int *)allocHandle.malloc(acc, length * sizeof(int));
            arC[id] = (int *)allocHandle.malloc(acc, length * sizeof(int));

            for(int i = 0; i < length; ++i)
            {
                arA[id][i] = static_cast<int>(id * length + i);
                arB[id][i] = static_cast<int>(id * length + i);
            }
        };
        const auto workDiv = alpaka::workdiv::WorkDivMembers<Dim, Idx>{
            Idx{grid}, Idx{block}, Idx{1}};
        alpaka::queue::enqueue(
            queue,
            alpaka::kernel::createTaskKernel<Acc>(
                workDiv,
                fillArrays,
                length,
                scatterAlloc.getAllocatorHandle()));
    }

    // add the 2 arrays (vector addition within each thread)
    // and do a thread-wise reduce to sums
    {
        auto sumsBufferAcc
            = alpaka::mem::buf::alloc<int, Idx>(dev, Idx{block * grid});

        auto addArrays = [] ALPAKA_FN_ACC(
                             const Acc & acc, int length, int * sums) {
            const auto id
                = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

            sums[id] = 0;
            for(int i = 0; i < length; ++i)
            {
                arC[id][i] = arA[id][i] + arB[id][i];
                sums[id] += arC[id][i];
            }
        };
        const auto workDiv = alpaka::workdiv::WorkDivMembers<Dim, Idx>{
            Idx{grid}, Idx{block}, Idx{1}};
        alpaka::queue::enqueue(
            queue,
            alpaka::kernel::createTaskKernel<Acc>(
                workDiv,
                addArrays,
                length,
                alpaka::mem::view::getPtrNative(sumsBufferAcc)));

        const auto hostDev = alpaka::pltf::getDevByIdx<alpaka::dev::DevCpu>(0);
        auto sumsBufferHost
            = alpaka::mem::buf::alloc<int, Idx>(hostDev, Idx{block * grid});
        alpaka::mem::view::copy(
            queue, sumsBufferHost, sumsBufferAcc, Idx{block * grid});
        alpaka::wait::wait(queue);

        const auto * sumsPtr = alpaka::mem::view::getPtrNative(sumsBufferHost);
        const auto sum
            = std::accumulate(sumsPtr, sumsPtr + block * grid, size_t{0});
        std::cout << "The sum of the arrays on GPU is " << sum << '\n';
    }

    const auto n = static_cast<size_t>(block * grid * length);
    const auto gaussian = n * (n - 1);
    std::cout << "The gaussian sum as comparison: " << gaussian << '\n';

    /*constexpr*/ if(mallocMC::Traits<ScatterAllocator>::providesAvailableSlots)
    {
        std::cout << "there are ";
        std::cout << scatterAlloc.getAvailableSlots(dev, queue, 1024U * 1024U);
        std::cout << " Slots of size 1MB available\n";
    }

    {
        auto freeArrays = [] ALPAKA_FN_ACC(
                              const Acc & acc,
                              ScatterAllocator::AllocatorHandle allocHandle) {
            const auto id
                = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
            allocHandle.free(acc, arA[id]);
            allocHandle.free(acc, arB[id]);
            allocHandle.free(acc, arC[id]);
        };
        const auto workDiv = alpaka::workdiv::WorkDivMembers<Dim, Idx>{
            Idx{grid}, Idx{block}, Idx{1}};
        alpaka::queue::enqueue(
            queue,
            alpaka::kernel::createTaskKernel<Acc>(
                workDiv, freeArrays, scatterAlloc.getAllocatorHandle()));
    }

    {
        auto freeArrayPointers
            = [] ALPAKA_FN_ACC(
                  const Acc & acc,
                  ScatterAllocator::AllocatorHandle allocHandle) {
                  allocHandle.free(acc, arA);
                  allocHandle.free(acc, arB);
                  allocHandle.free(acc, arC);
              };
        const auto workDiv
            = alpaka::workdiv::WorkDivMembers<Dim, Idx>{Idx{1}, Idx{1}, Idx{1}};
        alpaka::queue::enqueue(
            queue,
            alpaka::kernel::createTaskKernel<Acc>(
                workDiv, freeArrayPointers, scatterAlloc.getAllocatorHandle()));
    }

    return 0;
}