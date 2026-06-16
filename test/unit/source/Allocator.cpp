/*
  mallocMC: Memory Allocator for Many Core Architectures.

  Copyright 2025 Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Julian Johannes Lenz, Rene Widera

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

#include "mallocMC/allocator.hpp"

#include "mallocMC/alignmentPolicies/Shrink.hpp"
#include "mallocMC/creationPolicies/FlatterScatter.hpp"
#include "mallocMC/distributionPolicies/Noop.hpp"
#include "mallocMC/oOMPolicies/ReturnNull.hpp"
#include "mallocMC/reservePoolPolicies/AlpakaBuf.hpp"

#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <mallocMC/mallocMC.hpp>
using Dim = alpaka::DimInt<1>;
using Idx = std::size_t;

// Define the device accelerator
using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;

TEST_CASE("Allocator")
{
    SECTION("can be initialised with 0 memory.")
    {
        auto const platform = alpaka::Platform<Acc>{};
        auto const dev = alpaka::getDevByIdx(platform, 0);
        auto queue = alpaka::Queue<Acc, alpaka::Blocking>{dev};

        mallocMC::Allocator<
            alpaka::AccToTag<Acc>,
            mallocMC::CreationPolicies::FlatterScatter<>,
            mallocMC::DistributionPolicies::Noop,
            mallocMC::OOMPolicies::ReturnNull,
            mallocMC::ReservePoolPolicies::AlpakaBuf<Acc>,
            mallocMC::AlignmentPolicies::Shrink<>>
            allocator{dev, queue, 0};
    }
}
