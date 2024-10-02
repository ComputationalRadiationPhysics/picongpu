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
using Dim = alpaka::DimInt<1>;
using Acc = alpaka::AccGpuCudaRt<Dim, Idx>;

struct ScatterConfig
{
    static constexpr auto pagesize = 4096;
    static constexpr auto accessblocksize = 256U * 1024U;
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

template<typename ScatterAllocator>
void run()
{
    auto const platform = alpaka::Platform<Acc>{};
    const auto dev = alpaka::getDevByIdx(platform, 0);

    auto queue = alpaka::Queue<Acc, alpaka::Blocking>{dev};

    ScatterAllocator scatterAlloc(dev, queue, 1024U * 1024U); // 1 MiB
    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<Acc>(
            alpaka::WorkDivMembers<Dim, Idx>{Idx{1}, Idx{1}, Idx{1}},
            [] ALPAKA_FN_ACC(const Acc& acc, typename ScatterAllocator::AllocatorHandle allocHandle) {
                auto* ptr = allocHandle.malloc(acc, sizeof(int) * 1000);
                allocHandle.free(acc, ptr);
            },
            scatterAlloc.getAllocatorHandle()));
}

TEST_CASE("Scatter XMallocSIMD ReturnNull AlpakaBuf Shrink")
{
    using ScatterAllocator = mallocMC::Allocator<
        Acc,
        mallocMC::CreationPolicies::Scatter<ScatterConfig, ScatterHashParams>,
        mallocMC::DistributionPolicies::XMallocSIMD<DistributionConfig>,
        mallocMC::OOMPolicies::ReturnNull,
        mallocMC::ReservePoolPolicies::AlpakaBuf<Acc>,
        mallocMC::AlignmentPolicies::Shrink<AlignmentConfig>>;
    run<ScatterAllocator>();
}

TEST_CASE("Scatter XMallocSIMD ReturnNull AlpakaBuf Noop")
{
    using ScatterAllocator = mallocMC::Allocator<
        Acc,
        mallocMC::CreationPolicies::Scatter<ScatterConfig, ScatterHashParams>,
        mallocMC::DistributionPolicies::XMallocSIMD<DistributionConfig>,
        mallocMC::OOMPolicies::ReturnNull,
        mallocMC::ReservePoolPolicies::AlpakaBuf<Acc>,
        mallocMC::AlignmentPolicies::Noop>;
    run<ScatterAllocator>();
}

TEST_CASE("Scatter Noop ReturnNull AlpakaBuf Shrink")
{
    using ScatterAllocator = mallocMC::Allocator<
        Acc,
        mallocMC::CreationPolicies::Scatter<ScatterConfig, ScatterHashParams>,
        mallocMC::DistributionPolicies::Noop,
        mallocMC::OOMPolicies::ReturnNull,
        mallocMC::ReservePoolPolicies::AlpakaBuf<Acc>,
        mallocMC::AlignmentPolicies::Shrink<AlignmentConfig>>;
    run<ScatterAllocator>();
}

TEST_CASE("Scatter Noop ReturnNull AlpakaBuf Noop")
{
    using ScatterAllocator = mallocMC::Allocator<
        Acc,
        mallocMC::CreationPolicies::Scatter<ScatterConfig, ScatterHashParams>,
        mallocMC::DistributionPolicies::Noop,
        mallocMC::OOMPolicies::ReturnNull,
        mallocMC::ReservePoolPolicies::AlpakaBuf<Acc>,
        mallocMC::AlignmentPolicies::Noop>;
    run<ScatterAllocator>();
}

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
TEST_CASE("OldMalloc XMallocSIMD ReturnNull CudaSetLimits Shrink")
{
    using ScatterAllocator = mallocMC::Allocator<
        Acc,
        mallocMC::CreationPolicies::OldMalloc,
        mallocMC::DistributionPolicies::XMallocSIMD<DistributionConfig>,
        mallocMC::OOMPolicies::ReturnNull,
        mallocMC::ReservePoolPolicies::CudaSetLimits,
        mallocMC::AlignmentPolicies::Shrink<AlignmentConfig>>;
    run<ScatterAllocator>();

    cudaDeviceReset();
}

TEST_CASE("OldMalloc XMallocSIMD ReturnNull CudaSetLimits Noop")
{
    using ScatterAllocator = mallocMC::Allocator<
        Acc,
        mallocMC::CreationPolicies::OldMalloc,
        mallocMC::DistributionPolicies::XMallocSIMD<DistributionConfig>,
        mallocMC::OOMPolicies::ReturnNull,
        mallocMC::ReservePoolPolicies::CudaSetLimits,
        mallocMC::AlignmentPolicies::Noop>;
    run<ScatterAllocator>();

    cudaDeviceReset();
}

TEST_CASE("OldMalloc Noop ReturnNull CudaSetLimits Shrink")
{
    using ScatterAllocator = mallocMC::Allocator<
        Acc,
        mallocMC::CreationPolicies::OldMalloc,
        mallocMC::DistributionPolicies::Noop,
        mallocMC::OOMPolicies::ReturnNull,
        mallocMC::ReservePoolPolicies::CudaSetLimits,
        mallocMC::AlignmentPolicies::Shrink<AlignmentConfig>>;
    run<ScatterAllocator>();

    cudaDeviceReset();
}

TEST_CASE("OldMalloc Noop ReturnNull CudaSetLimits Noop")
{
    using ScatterAllocator = mallocMC::Allocator<
        Acc,
        mallocMC::CreationPolicies::OldMalloc,
        mallocMC::DistributionPolicies::Noop,
        mallocMC::OOMPolicies::ReturnNull,
        mallocMC::ReservePoolPolicies::CudaSetLimits,
        mallocMC::AlignmentPolicies::Noop>;
    run<ScatterAllocator>();

    cudaDeviceReset();
}
#endif
