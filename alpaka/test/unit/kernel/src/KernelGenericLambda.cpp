/* Copyright 2019 Axel Huebl, Benjamin Worpitz, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/kernel/Traits.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch.hpp>

// CUDA C Programming guide says: "__host__ __device__ extended lambdas cannot be generic lambdas"
#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED)

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE("genericLambdaKernelIsWorking", "[kernel]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    auto kernel = [] ALPAKA_FN_ACC(auto const& acc, bool* success) -> void {
        ALPAKA_CHECK(
            *success,
            static_cast<alpaka::Idx<Acc>>(1) == (alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)).prod());
    };

    REQUIRE(fixture(kernel));
}

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE("variadicGenericLambdaKernelIsWorking", "[kernel]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    std::uint32_t const arg1 = 42u;
    std::uint32_t const arg2 = 43u;
    auto kernel = [] ALPAKA_FN_ACC(Acc const& acc, bool* success, auto... args) -> void {
        alpaka::ignore_unused(acc);

        ALPAKA_CHECK(*success, alpaka::meta::foldr([](auto a, auto b) { return a + b; }, args...) == (42u + 43u));
    };

    REQUIRE(fixture(kernel, arg1, arg2));
}

#endif
