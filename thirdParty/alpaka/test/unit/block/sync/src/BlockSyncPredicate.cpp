/**
 * \file
 * Copyright 2017 Benjamin Worpitz
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

#include <boost/assert.hpp>
#include <boost/predef.h>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include <boost/test/unit_test.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif

//#############################################################################
class BlockSyncPredicateTestKernel
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc) const
    -> void
    {
        using Size = alpaka::size::Size<TAcc>;

        // Get the index of the current thread within the block and the block extent and map them to 1D.
        auto const blockThreadIdx(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc));
        auto const blockThreadExtent(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc));
        auto const blockThreadIdx1D(alpaka::idx::mapIdx<1u>(blockThreadIdx, blockThreadExtent)[0u]);
        auto const blockThreadExtent1D(blockThreadExtent.prod());

        // syncBlockThreadsPredicate<alpaka::block::sync::op::Count>
        {
            Size const modulus(2u);
            int const predicate(static_cast<int>(blockThreadIdx1D % modulus));
            auto const result(alpaka::block::sync::syncBlockThreadsPredicate<alpaka::block::sync::op::Count>(acc, predicate));
            auto const expectedResult(static_cast<int>(blockThreadExtent1D / modulus));
            BOOST_VERIFY(expectedResult == result);
        }
        {
            Size const modulus(3u);
            int const predicate(static_cast<int>(blockThreadIdx1D % modulus));
            auto const result(alpaka::block::sync::syncBlockThreadsPredicate<alpaka::block::sync::op::Count>(acc, predicate));
            auto const expectedResult(static_cast<int>(blockThreadExtent1D - ((blockThreadExtent1D + modulus - static_cast<Size>(1u)) / modulus)));
            BOOST_VERIFY(expectedResult == result);
        }

        // syncBlockThreadsPredicate<alpaka::block::sync::op::LogicalAnd>
        {
            int const predicate(1);
            auto const result(alpaka::block::sync::syncBlockThreadsPredicate<alpaka::block::sync::op::LogicalAnd>(acc, predicate));
            BOOST_VERIFY(result == 1);
        }
        {
            int const predicate(0);
            auto const result(alpaka::block::sync::syncBlockThreadsPredicate<alpaka::block::sync::op::LogicalAnd>(acc, predicate));
            BOOST_VERIFY(result == 0);
        }
        {
            int const predicate(blockThreadIdx1D != 0);
            auto const result(alpaka::block::sync::syncBlockThreadsPredicate<alpaka::block::sync::op::LogicalAnd>(acc, predicate));
            BOOST_VERIFY(result == 0);
        }

        // syncBlockThreadsPredicate<alpaka::block::sync::op::LogicalOr>
        {
            int const predicate(1);
            auto const result(alpaka::block::sync::syncBlockThreadsPredicate<alpaka::block::sync::op::LogicalOr>(acc, predicate));
            BOOST_VERIFY(result == 1);
        }
        {
            int const predicate(0);
            auto const result(alpaka::block::sync::syncBlockThreadsPredicate<alpaka::block::sync::op::LogicalOr>(acc, predicate));
            BOOST_VERIFY(result == 0);
        }
        {
            int const predicate(static_cast<int>(blockThreadIdx1D != 1));
            auto const result(alpaka::block::sync::syncBlockThreadsPredicate<alpaka::block::sync::op::LogicalOr>(acc, predicate));
            BOOST_VERIFY(result == 1);
        }
    }
};

BOOST_AUTO_TEST_SUITE(blockSync)

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    synchronizePredicate,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Size = alpaka::size::Size<TAcc>;

    BlockSyncPredicateTestKernel kernel;

    // 4^Dim
    {
        alpaka::test::KernelExecutionFixture<TAcc> fixture(
            alpaka::vec::Vec<Dim, Size>::all(static_cast<Size>(4u)));

        BOOST_REQUIRE_EQUAL(
            true,
            fixture(
                kernel));
    }

    // 1^Dim
    {
        alpaka::test::KernelExecutionFixture<TAcc> fixture(
            alpaka::vec::Vec<Dim, Size>::ones());

        BOOST_REQUIRE_EQUAL(
            true,
            fixture(
                kernel));
    }
}

BOOST_AUTO_TEST_SUITE_END()
