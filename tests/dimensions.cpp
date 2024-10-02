/*
  mallocMC: Memory Allocator for Many Core Architectures.

  Copyright 2020 Helmholtz-Zentrum Dresden - Rossendorf,
                 CERN

  Author(s):  Bernhard Manfred Gruber

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
#include <catch2/catch.hpp>
#include <mallocMC/mallocMC.hpp>

using Idx = std::size_t;

struct ScatterConfig
{
    static constexpr auto pagesize = 4096;
    static constexpr auto accessblocksize = 256u * 1024u;
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

struct DistributionConfig
{
    static constexpr auto pagesize = ScatterConfig::pagesize;
};

struct AlignmentConfig
{
    static constexpr auto dataAlignment = 16;
};

ALPAKA_STATIC_ACC_MEM_GLOBAL int** deviceArray;

template<template<typename, typename> typename AccTemplate>
void test1D()
{
    using Dim = alpaka::DimInt<1>;
    using Acc = AccTemplate<Dim, Idx>;

    using ScatterAllocator = mallocMC::Allocator<
        Acc,
        mallocMC::CreationPolicies::Scatter<ScatterConfig, ScatterHashParams>,
        // mallocMC::CreationPolicies::OldMalloc,
        mallocMC::DistributionPolicies::Noop,
        mallocMC::OOMPolicies::ReturnNull,
        mallocMC::ReservePoolPolicies::AlpakaBuf<Acc>,
        // mallocMC::ReservePoolPolicies::CudaSetLimits,
        mallocMC::AlignmentPolicies::Shrink<AlignmentConfig>>;

    auto const platform = alpaka::Platform<Acc>{};
    const auto dev = alpaka::getDevByIdx(platform, 0);
    auto queue = alpaka::Queue<Acc, alpaka::Blocking>{dev};

    constexpr auto N = 16;
    static_assert(N <= mallocMC::maxThreadsPerBlock, "");

    ScatterAllocator scatterAlloc(dev, queue, 1024U * 1024U); // 1 MiB

    // make 1 allocation from 1 thread for N * N pointers
    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<Acc>(
            alpaka::WorkDivMembers<Dim, Idx>{Idx{1}, Idx{1}, Idx{1}},
            [] ALPAKA_FN_ACC(const Acc& acc, int dim, typename ScatterAllocator::AllocatorHandle allocHandle) {
                deviceArray = (int**) allocHandle.malloc(acc, sizeof(int*) * dim * dim);
            },
            N,
            scatterAlloc.getAllocatorHandle()));

    // make N * N allocations from N block of N threads for ints
    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<Acc>(
            alpaka::WorkDivMembers<Dim, Idx>{Idx{N}, Idx{N}, Idx{1}},
            [] ALPAKA_FN_ACC(const Acc& acc, typename ScatterAllocator::AllocatorHandle allocHandle) {
                const auto i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
                deviceArray[i] = (int*) allocHandle.malloc(acc, sizeof(int));
            },
            scatterAlloc.getAllocatorHandle()));

    const auto slots = scatterAlloc.getAvailableSlots(dev, queue, sizeof(int));
    const auto heapInfo = scatterAlloc.getHeapLocations().at(0);
    std::cout << alpaka::trait::GetAccName<Acc>::getAccName() << " slots: " << slots << " heap size: " << heapInfo.size
              << '\n';

    // free N * N allocations from N block of N threads for ints
    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<Acc>(
            alpaka::WorkDivMembers<Dim, Idx>{Idx{N}, Idx{N}, Idx{1}},
            [] ALPAKA_FN_ACC(const Acc& acc, typename ScatterAllocator::AllocatorHandle allocHandle) {
                const auto i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
                allocHandle.free(acc, deviceArray[i]);
            },
            scatterAlloc.getAllocatorHandle()));

    // free 1 allocation from 1 thread for N * N pointers
    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<Acc>(
            alpaka::WorkDivMembers<Dim, Idx>{Idx{1}, Idx{1}, Idx{1}},
            [] ALPAKA_FN_ACC(const Acc& acc, typename ScatterAllocator::AllocatorHandle allocHandle) {
                allocHandle.free(acc, deviceArray);
            },
            scatterAlloc.getAllocatorHandle()));
}

template<template<typename, typename> typename AccTemplate>
void test2D()
{
    using Dim = alpaka::DimInt<2>;
    using Acc = AccTemplate<Dim, Idx>;

    using ScatterAllocator = mallocMC::Allocator<
        Acc,
        mallocMC::CreationPolicies::Scatter<ScatterConfig, ScatterHashParams>,
        mallocMC::DistributionPolicies::XMallocSIMD<DistributionConfig>,
        mallocMC::OOMPolicies::ReturnNull,
        mallocMC::ReservePoolPolicies::AlpakaBuf<Acc>,
        mallocMC::AlignmentPolicies::Shrink<AlignmentConfig>>;

    auto const platform = alpaka::Platform<Acc>{};
    const auto dev = alpaka::getDevByIdx(platform, 0);
    auto queue = alpaka::Queue<Acc, alpaka::Blocking>{dev};

    constexpr auto N = 8;
    static_assert(N * N <= mallocMC::maxThreadsPerBlock, "");

    ScatterAllocator scatterAlloc(dev, queue, 1024U * 1024U); // 1 MiB

    // make 1 allocation from 1 thread for N*N * N*N pointers
    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<Acc>(
            alpaka::WorkDivMembers<Dim, Idx>{
                alpaka::Vec<Dim, Idx>::all(1),
                alpaka::Vec<Dim, Idx>::all(1),
                alpaka::Vec<Dim, Idx>::all(1)},
            [] ALPAKA_FN_ACC(const Acc& acc, int dim, typename ScatterAllocator::AllocatorHandle allocHandle) {
                deviceArray = (int**) allocHandle.malloc(acc, sizeof(int*) * dim * dim * dim * dim);
            },
            N,
            scatterAlloc.getAllocatorHandle()));

    // make N*N * N*N allocations from N*N block of N*N threads for ints
    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<Acc>(
            alpaka::WorkDivMembers<Dim, Idx>{
                alpaka::Vec<Dim, Idx>::all(N),
                alpaka::Vec<Dim, Idx>::all(N),
                alpaka::Vec<Dim, Idx>::all(1)},
            [] ALPAKA_FN_ACC(const Acc& acc, int dim, typename ScatterAllocator::AllocatorHandle allocHandle) {
                const auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
                deviceArray[idx[0] * dim * dim + idx[1]] = (int*) allocHandle.malloc(acc, sizeof(int));
            },
            N,
            scatterAlloc.getAllocatorHandle()));

    const auto slots = scatterAlloc.getAvailableSlots(dev, queue, sizeof(int));
    const auto heapInfo = scatterAlloc.getHeapLocations().at(0);
    std::cout << alpaka::trait::GetAccName<Acc>::getAccName() << " slots: " << slots << " heap size: " << heapInfo.size
              << '\n';

    // free N*N * N*N allocations from N*N block of N*N threads for ints
    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<Acc>(
            alpaka::WorkDivMembers<Dim, Idx>{
                alpaka::Vec<Dim, Idx>::all(N),
                alpaka::Vec<Dim, Idx>::all(N),
                alpaka::Vec<Dim, Idx>::all(1)},
            [] ALPAKA_FN_ACC(const Acc& acc, int dim, typename ScatterAllocator::AllocatorHandle allocHandle) {
                const auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
                allocHandle.free(acc, deviceArray[idx[0] * dim * dim + idx[1]]);
            },
            N,
            scatterAlloc.getAllocatorHandle()));

    // free 1 allocation from 1 thread for N*N * N*N pointers
    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<Acc>(
            alpaka::WorkDivMembers<Dim, Idx>{
                alpaka::Vec<Dim, Idx>::all(1),
                alpaka::Vec<Dim, Idx>::all(1),
                alpaka::Vec<Dim, Idx>::all(1)},
            [] ALPAKA_FN_ACC(const Acc& acc, typename ScatterAllocator::AllocatorHandle allocHandle) {
                allocHandle.free(acc, deviceArray);
            },
            scatterAlloc.getAllocatorHandle()));
}

template<template<typename, typename> typename AccTemplate>
void test3D()
{
    using Dim = alpaka::DimInt<3>;
    using Acc = AccTemplate<Dim, Idx>;

    using ScatterAllocator = mallocMC::Allocator<
        Acc,
        mallocMC::CreationPolicies::Scatter<ScatterConfig, ScatterHashParams>,
        mallocMC::DistributionPolicies::XMallocSIMD<DistributionConfig>,
        mallocMC::OOMPolicies::ReturnNull,
        mallocMC::ReservePoolPolicies::AlpakaBuf<Acc>,
        mallocMC::AlignmentPolicies::Shrink<AlignmentConfig>>;

    auto const platform = alpaka::Platform<Acc>{};
    const auto dev = alpaka::getDevByIdx(platform, 0);
    auto queue = alpaka::Queue<Acc, alpaka::Blocking>{dev};

    constexpr auto N = 4;
    static_assert(N * N * N <= mallocMC::maxThreadsPerBlock, "");

    ScatterAllocator scatterAlloc(dev, queue, 1024U * 1024U); // 1 MiB

    // make 1 allocation from 1 thread for N*N*N * N*N*N pointers
    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<Acc>(
            alpaka::WorkDivMembers<Dim, Idx>{
                alpaka::Vec<Dim, Idx>::all(1),
                alpaka::Vec<Dim, Idx>::all(1),
                alpaka::Vec<Dim, Idx>::all(1)},
            [] ALPAKA_FN_ACC(const Acc& acc, int dim, typename ScatterAllocator::AllocatorHandle allocHandle) {
                deviceArray = (int**) allocHandle.malloc(acc, sizeof(int*) * dim * dim * dim * dim * dim * dim);
            },
            N,
            scatterAlloc.getAllocatorHandle()));

    // make N*N*N * N*N*N allocations from N*N*N blocks of N*N*N threads for
    // ints
    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<Acc>(
            alpaka::WorkDivMembers<Dim, Idx>{
                alpaka::Vec<Dim, Idx>::all(N),
                alpaka::Vec<Dim, Idx>::all(N),
                alpaka::Vec<Dim, Idx>::all(1)},
            [] ALPAKA_FN_ACC(const Acc& acc, int dim, typename ScatterAllocator::AllocatorHandle allocHandle) {
                const auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
                deviceArray[idx[0] * dim * dim * dim * dim + idx[1] * dim * dim + idx[0]]
                    = (int*) allocHandle.malloc(acc, sizeof(int));
            },
            N,
            scatterAlloc.getAllocatorHandle()));

    const auto slots = scatterAlloc.getAvailableSlots(dev, queue, sizeof(int));
    const auto heapInfo = scatterAlloc.getHeapLocations().at(0);
    std::cout << alpaka::trait::GetAccName<Acc>::getAccName() << " slots: " << slots << " heap size: " << heapInfo.size
              << '\n';

    // free N*N*N * N*N*N allocations from N*N*N blocks of N*N*N threads for
    // ints
    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<Acc>(
            alpaka::WorkDivMembers<Dim, Idx>{
                alpaka::Vec<Dim, Idx>::all(N),
                alpaka::Vec<Dim, Idx>::all(N),
                alpaka::Vec<Dim, Idx>::all(1)},
            [] ALPAKA_FN_ACC(const Acc& acc, int dim, typename ScatterAllocator::AllocatorHandle allocHandle) {
                const auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
                allocHandle.free(acc, deviceArray[idx[0] * dim * dim * dim * dim + idx[1] * dim * dim + idx[0]]);
            },
            N,
            scatterAlloc.getAllocatorHandle()));

    // free 1 allocation from 1 thread for N*N*N * N*N*N pointers
    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<Acc>(
            alpaka::WorkDivMembers<Dim, Idx>{
                alpaka::Vec<Dim, Idx>::all(1),
                alpaka::Vec<Dim, Idx>::all(1),
                alpaka::Vec<Dim, Idx>::all(1)},
            [] ALPAKA_FN_ACC(const Acc& acc, typename ScatterAllocator::AllocatorHandle allocHandle) {
                allocHandle.free(acc, deviceArray);
            },
            scatterAlloc.getAllocatorHandle()));
}

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
TEST_CASE("1D AccGpuCudaRt")
{
    test1D<alpaka::AccGpuCudaRt>();
}

TEST_CASE("2D AccGpuCudaRt")
{
    test2D<alpaka::AccGpuCudaRt>();
}

TEST_CASE("3D AccGpuCudaRt")
{
    test3D<alpaka::AccGpuCudaRt>();
}
#endif

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
TEST_CASE("1D AccGpuHipRt")
{
    test1D<alpaka::AccGpuHipRt>();
}

TEST_CASE("2D AccGpuHipRt")
{
    test2D<alpaka::AccGpuHipRt>();
}

TEST_CASE("3D AccGpuHipRt")
{
    test3D<alpaka::AccGpuHipRt>();
}
#endif

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED)
TEST_CASE("1D AccCpuThreads")
{
    test1D<alpaka::AccCpuThreads>();
}

TEST_CASE("2D AccCpuThreads")
{
    test2D<alpaka::AccCpuThreads>();
}

TEST_CASE("3D AccCpuThreads")
{
    test3D<alpaka::AccCpuThreads>();
}
#endif

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED)
TEST_CASE("1D AccCpuOmp2Threads")
{
    test1D<alpaka::AccCpuOmp2Threads>();
}

TEST_CASE("2D AccCpuOmp2Threads")
{
    test2D<alpaka::AccCpuOmp2Threads>();
}

TEST_CASE("3D AccCpuOmp2Threads")
{
    test3D<alpaka::AccCpuOmp2Threads>();
}
#endif

#if defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
TEST_CASE("1D AccCpuOmp2Blocks")
{
    test1D<alpaka::AccCpuOmp2Blocks>();
}

TEST_CASE("2D AccCpuOmp2Blocks")
{
    test2D<alpaka::AccCpuOmp2Blocks>();
}

TEST_CASE("3D AccCpuOmp2Blocks")
{
    test3D<alpaka::AccCpuOmp2Blocks>();
}
#endif

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
TEST_CASE("1D AccCpuSerial")
{
    test1D<alpaka::AccCpuSerial>();
}

TEST_CASE("2D AccCpuSerial")
{
    test2D<alpaka::AccCpuSerial>();
}

TEST_CASE("3D AccCpuSerial")
{
    test3D<alpaka::AccCpuSerial>();
}
#endif
