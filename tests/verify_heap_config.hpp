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

#pragma once

#include <alpaka/alpaka.hpp>
#include <mallocMC/mallocMC.hpp>

using Dim = alpaka::DimInt<1>;
using Idx = std::size_t;
// using Acc = alpaka::AccCpuThreads<Dim, Idx>;
// using Acc = alpaka::AccCpuOmp2Threads<Dim, Idx>;
using Acc = alpaka::AccGpuCudaRt<Dim, Idx>;

// configurate the CreationPolicy "Scatter"
struct ScatterConfig
{
    static constexpr auto pagesize = 4096;
    static constexpr auto accessblocks = 8;
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

// configure the DistributionPolicy "XMallocSIMD"
struct DistributionConfig
{
    static constexpr auto pagesize = ScatterConfig::pagesize;
};

// configure the AlignmentPolicy "Shrink"
struct AlignmentConfig
{
    static constexpr auto dataAlignment = 16;
};

// Define a new allocator and call it ScatterAllocator
// which resembles the behaviour of ScatterAlloc
using ScatterAllocator = mallocMC::Allocator<
    Acc,
    mallocMC::CreationPolicies::Scatter<ScatterConfig, ScatterHashParams>,
    mallocMC::DistributionPolicies::XMallocSIMD<DistributionConfig>,
    mallocMC::OOMPolicies::ReturnNull,
    mallocMC::ReservePoolPolicies::AlpakaBuf<Acc>,
    mallocMC::AlignmentPolicies::Shrink<AlignmentConfig>>;
