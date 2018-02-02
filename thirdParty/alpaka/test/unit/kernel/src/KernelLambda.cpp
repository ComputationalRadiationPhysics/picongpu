/**
 * \file
 * Copyright 2015 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * alpaka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * alpaka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with alpaka.
 * If not, see <http://www.gnu.org/licenses/>.
 */

// \Hack: Boost.MPL defines BOOST_MPL_CFG_GPU_ENABLED to __host__ __device__ if nvcc is used.
// BOOST_AUTO_TEST_CASE_TEMPLATE and its internals are not GPU enabled but is using boost::mpl::for_each internally.
// For each template parameter this leads to:
// /home/travis/build/boost/boost/mpl/for_each.hpp(78): warning: calling a __host__ function from a __host__ __device__ function is not allowed
// because boost::mpl::for_each has the BOOST_MPL_CFG_GPU_ENABLED attribute but the test internals are pure host methods.
// Because we do not use MPL within GPU code here, we can disable the MPL GPU support.
#define BOOST_MPL_CFG_GPU_ENABLED

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/Acc.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>

#include <boost/predef.h>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include <boost/test/unit_test.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif

BOOST_AUTO_TEST_SUITE(kernel)

// nvcc < 7.5 does not support lambdas as kernels.
#if !BOOST_COMP_NVCC || BOOST_COMP_NVCC >= BOOST_VERSION_NUMBER(7, 5, 0)
// nvcc 7.5 does not support heterogeneous lambdas (__host__ __device__) as kernels but only __device__ lambdas.
// So with nvcc 7.5 this only works in CUDA only mode or by using ALPAKA_FN_ACC_CUDA_ONLY instead of ALPAKA_FN_ACC
#if !BOOST_COMP_NVCC || BOOST_COMP_NVCC >= BOOST_VERSION_NUMBER(8, 0, 0) || defined(ALPAKA_ACC_GPU_CUDA_ONLY_MODE)

// clang prior to 4.0.0 did not support the __host__ __device__ attributes at the nonstandard position between [] and () but only after ().
// See: https://llvm.org/bugs/show_bug.cgi?id=26341
#if !BOOST_COMP_CLANG_CUDA || BOOST_COMP_CLANG_CUDA >= BOOST_VERSION_NUMBER(4, 0, 0)

#if !ALPAKA_CI
//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    lambdaKernelIsWorking,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Size = alpaka::size::Size<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Size>::ones());

    auto kernel =
        [] ALPAKA_FN_ACC (TAcc const & acc)
        -> void
        {
            // Do something useless on the accelerator.
            alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc);
        };

    BOOST_REQUIRE_EQUAL(
        true,
        fixture(
            kernel));
}
#endif

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    lambdaKernelWithArgumentIsWorking,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Size = alpaka::size::Size<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Size>::ones());

    std::uint32_t const arg = 42u;
    auto kernel =
        [] ALPAKA_FN_ACC (TAcc const & acc, std::uint32_t const & arg1)
        -> void
        {
            // Do something useless on the accelerator.
            alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc);

            BOOST_VERIFY(42u == arg1);
        };

    BOOST_REQUIRE_EQUAL(
        true,
        fixture(
            kernel,
            arg));
}

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    lambdaKernelWithCapturingIsWorking,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Size = alpaka::size::Size<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Size>::ones());

    std::uint32_t const arg = 42u;
    auto kernel =
        [arg] ALPAKA_FN_ACC (TAcc const & acc)
        -> void
        {
            // Do something useless on the accelerator.
            alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc);

            (void)arg;
            BOOST_VERIFY(42u == arg);
        };

    BOOST_REQUIRE_EQUAL(
        true,
        fixture(
            kernel));
}

// Generic lambdas are a C++14 feature.
#if !defined(BOOST_NO_CXX14_GENERIC_LAMBDAS)
#if !ALPAKA_CI
//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    genericLambdaKernelIsWorking,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Size = alpaka::size::Size<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Size>::ones());

    auto kernel =
        [] ALPAKA_FN_ACC (auto const & acc)
        -> void
        {
            // Do something useless on the accelerator.
            alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc);
        };

    BOOST_REQUIRE_EQUAL(
        true,
        fixture(
            kernel));
}
#endif

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    variadicGenericLambdaKernelIsWorking,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Size = alpaka::size::Size<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Size>::ones());

    std::uint32_t const arg1 = 42u;
    std::uint32_t const arg2 = 43u;
    auto kernel =
        [] ALPAKA_FN_ACC (TAcc const & acc, auto ... args)
        -> void
        {
            // Do something useless on the accelerator.
            alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc);

            BOOST_VERIFY(alpaka::meta::foldr([](auto a, auto b){return a + b;}, args...) == (42u + 43u));
        };

    BOOST_REQUIRE_EQUAL(
        true,
        fixture(
            kernel,
            arg1,
            arg2));
}
#endif

#endif

#endif

#endif

BOOST_AUTO_TEST_SUITE_END()
