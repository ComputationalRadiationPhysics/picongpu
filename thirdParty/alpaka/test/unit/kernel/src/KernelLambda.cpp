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

// NVCC needs --expt-extended-lambda
#if !defined(__NVCC__) || \
    ( defined(__NVCC__) && defined(__CUDACC_EXTENDED_LAMBDA__) )

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

#include <alpaka/core/BoostPredef.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include <boost/test/unit_test.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif

BOOST_AUTO_TEST_SUITE(kernel)

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    lambdaKernelIsWorking,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    auto kernel =
        [] ALPAKA_FN_ACC (
            TAcc const & acc,
            bool * success)
        -> void
        {
            ALPAKA_CHECK(
                *success,
                static_cast<alpaka::idx::Idx<TAcc>>(1) == (alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)).prod());
        };

    BOOST_REQUIRE_EQUAL(
        true,
        fixture(
            kernel));
}

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    lambdaKernelWithArgumentIsWorking,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    std::uint32_t const arg = 42u;
    auto kernel =
        [] ALPAKA_FN_ACC (
            TAcc const & acc,
            bool * success,
            std::uint32_t const & arg1)
        -> void
        {
            alpaka::ignore_unused(acc);

            ALPAKA_CHECK(*success, 42u == arg1);
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
    using Idx = alpaka::idx::Idx<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    std::uint32_t const arg = 42u;

#if BOOST_COMP_CLANG >= BOOST_VERSION_NUMBER(5,0,0)
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-lambda-capture"
#endif
    auto kernel =
        [arg] ALPAKA_FN_ACC (
            TAcc const & acc,
            bool * success)
        -> void
        {
            alpaka::ignore_unused(acc);

            ALPAKA_CHECK(*success, 42u == arg);
        };
#if BOOST_COMP_CLANG >= BOOST_VERSION_NUMBER(5,0,0)
    #pragma clang diagnostic pop
#endif

    BOOST_REQUIRE_EQUAL(
        true,
        fixture(
            kernel));
}

BOOST_AUTO_TEST_SUITE_END()

#endif
