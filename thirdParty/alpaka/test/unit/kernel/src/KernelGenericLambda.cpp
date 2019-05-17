/* Copyright 2019 Axel Huebl, Benjamin Worpitz, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/Acc.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>

#include <catch2/catch.hpp>

// Generic lambdas are a C++14 feature.
#if !defined(BOOST_NO_CXX14_GENERIC_LAMBDAS)
// CUDA C Programming guide says: "__host__ __device__ extended lambdas cannot be generic lambdas"
#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
//-----------------------------------------------------------------------------
struct TestTemplateGeneric
{
template< typename TAcc >
void operator()()
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    auto kernel =
        [] ALPAKA_FN_ACC (
            auto const & acc,
            bool * success)
        -> void
        {
            ALPAKA_CHECK(
                *success,
                static_cast<alpaka::idx::Idx<TAcc>>(1) == (alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)).prod());
        };

    REQUIRE(fixture(kernel));
}
};

//-----------------------------------------------------------------------------
struct TestTemplateVariadic
{
template< typename TAcc >
void operator()()
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    std::uint32_t const arg1 = 42u;
    std::uint32_t const arg2 = 43u;
    auto kernel =
        [] ALPAKA_FN_ACC (
            TAcc const & acc,
            bool * success,
            auto ... args)
        -> void
        {
            alpaka::ignore_unused(acc);

            ALPAKA_CHECK(
                *success,
                alpaka::meta::foldr([](auto a, auto b){return a + b;}, args...) == (42u + 43u));
        };

    REQUIRE(fixture(kernel, arg1, arg2));
}
};

TEST_CASE( "genericLambdaKernelIsWorking", "[kernel]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplateGeneric() );
}

TEST_CASE( "variadicGenericLambdaKernelIsWorking", "[kernel]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplateVariadic() );
}

#endif
#endif
