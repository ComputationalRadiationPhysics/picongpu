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
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/test/Array.hpp>
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

BOOST_AUTO_TEST_SUITE(blockSharedMemSt)

//#############################################################################
class BlockSharedMemStNonNullTestKernel
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        bool * success) const
    -> void
    {
#if BOOST_COMP_GNUC >= BOOST_VERSION_NUMBER(6, 0, 0)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Waddress"  // warning: the compiler can assume that the address of ‘a’ will never be NULL [-Waddress]
#endif
        // Multiple runs to make sure it really works.
        for(std::size_t i=0u; i<10; ++i)
        {
            auto && a = alpaka::block::shared::st::allocVar<std::uint32_t, __COUNTER__>(acc);
            ALPAKA_CHECK(*success, static_cast<std::uint32_t *>(nullptr) != &a);

            auto && b = alpaka::block::shared::st::allocVar<std::uint32_t, __COUNTER__>(acc);
            ALPAKA_CHECK(*success, static_cast<std::uint32_t *>(nullptr) != &b);

            auto && c = alpaka::block::shared::st::allocVar<float, __COUNTER__>(acc);
            ALPAKA_CHECK(*success, static_cast<float *>(nullptr) != &c);

            auto && d = alpaka::block::shared::st::allocVar<double, __COUNTER__>(acc);
            ALPAKA_CHECK(*success, static_cast<double *>(nullptr) != &d);

            auto && e = alpaka::block::shared::st::allocVar<std::uint64_t, __COUNTER__>(acc);
            ALPAKA_CHECK(*success, static_cast<std::uint64_t *>(nullptr) != &e);


            auto && f = alpaka::block::shared::st::allocVar<alpaka::test::Array<std::uint32_t, 32>, __COUNTER__>(acc);
            ALPAKA_CHECK(*success, static_cast<std::uint32_t *>(nullptr) != &f[0]);

            auto && g = alpaka::block::shared::st::allocVar<alpaka::test::Array<std::uint32_t, 32>, __COUNTER__>(acc);
            ALPAKA_CHECK(*success, static_cast<std::uint32_t *>(nullptr) != &g[0]);

            auto && h = alpaka::block::shared::st::allocVar<alpaka::test::Array<double, 16>, __COUNTER__>(acc);
            ALPAKA_CHECK(*success, static_cast<double *>(nullptr) != &h[0]);
        }
#if BOOST_COMP_GNUC >= BOOST_VERSION_NUMBER(6, 0, 0)
    #pragma GCC diagnostic pop
#endif
    }
};

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    nonNull,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    // Use multiple threads to make sure the synchronization really works.
    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Idx>::all(static_cast<Idx>(3u)));

    BlockSharedMemStNonNullTestKernel kernel;

    BOOST_REQUIRE_EQUAL(
        true,
        fixture(
            kernel));
}

//#############################################################################
class BlockSharedMemStSameTypeDifferentAdressTestKernel
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        bool * success) const
    -> void
    {
        // Multiple runs to make sure it really works.
        for(std::size_t i=0u; i<10; ++i)
        {
            auto && a = alpaka::block::shared::st::allocVar<std::uint32_t, __COUNTER__>(acc);
            auto && b = alpaka::block::shared::st::allocVar<std::uint32_t, __COUNTER__>(acc);
            ALPAKA_CHECK(*success, &a != &b);
            auto && c = alpaka::block::shared::st::allocVar<std::uint32_t, __COUNTER__>(acc);
            ALPAKA_CHECK(*success, &b != &c);
            ALPAKA_CHECK(*success, &a != &c);
            ALPAKA_CHECK(*success, &b != &c);

            auto && d = alpaka::block::shared::st::allocVar<alpaka::test::Array<std::uint32_t, 32>, __COUNTER__>(acc);
            ALPAKA_CHECK(*success, &a != &d[0]);
            ALPAKA_CHECK(*success, &b != &d[0]);
            ALPAKA_CHECK(*success, &c != &d[0]);
            auto && e = alpaka::block::shared::st::allocVar<alpaka::test::Array<std::uint32_t, 32>, __COUNTER__>(acc);
            ALPAKA_CHECK(*success, &a != &e[0]);
            ALPAKA_CHECK(*success, &b != &e[0]);
            ALPAKA_CHECK(*success, &c != &e[0]);
            ALPAKA_CHECK(*success, &d[0] != &e[0]);
        }
    }
};

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    sameTypeDifferentAddress,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    // Use multiple threads to make sure the synchronization really works.
    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Idx>::all(static_cast<Idx>(3u)));

    BlockSharedMemStSameTypeDifferentAdressTestKernel kernel;

    BOOST_REQUIRE_EQUAL(
        true,
        fixture(
            kernel));
}

BOOST_AUTO_TEST_SUITE_END()
