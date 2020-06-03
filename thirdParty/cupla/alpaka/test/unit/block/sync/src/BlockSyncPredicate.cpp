/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/block/sync/Traits.hpp>

#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>

#include <catch2/catch.hpp>

//#############################################################################
class BlockSyncPredicateTestKernel
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
        using Idx = alpaka::idx::Idx<TAcc>;

        // Get the index of the current thread within the block and the block extent and map them to 1D.
        auto const blockThreadIdx(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc));
        auto const blockThreadExtent(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc));
        auto const blockThreadIdx1D(alpaka::idx::mapIdx<1u>(blockThreadIdx, blockThreadExtent)[0u]);
        auto const blockThreadExtent1D(blockThreadExtent.prod());

        // syncBlockThreadsPredicate<alpaka::block::sync::op::Count>
        {
            Idx const modulus(2u);
            int const predicate(static_cast<int>(blockThreadIdx1D % modulus));
            auto const result(alpaka::block::sync::syncBlockThreadsPredicate<alpaka::block::sync::op::Count>(acc, predicate));
            auto const expectedResult(static_cast<int>(blockThreadExtent1D / modulus));
            ALPAKA_CHECK(*success, expectedResult == result);
        }
        {
            Idx const modulus(3u);
            int const predicate(static_cast<int>(blockThreadIdx1D % modulus));
            auto const result(alpaka::block::sync::syncBlockThreadsPredicate<alpaka::block::sync::op::Count>(acc, predicate));
            auto const expectedResult(static_cast<int>(blockThreadExtent1D - ((blockThreadExtent1D + modulus - static_cast<Idx>(1u)) / modulus)));
            ALPAKA_CHECK(*success, expectedResult == result);
        }

        // syncBlockThreadsPredicate<alpaka::block::sync::op::LogicalAnd>
        {
            int const predicate(1);
            auto const result(alpaka::block::sync::syncBlockThreadsPredicate<alpaka::block::sync::op::LogicalAnd>(acc, predicate));
            ALPAKA_CHECK(*success, result == 1);
        }
        {
            int const predicate(0);
            auto const result(alpaka::block::sync::syncBlockThreadsPredicate<alpaka::block::sync::op::LogicalAnd>(acc, predicate));
            ALPAKA_CHECK(*success, result == 0);
        }
        {
            int const predicate(blockThreadIdx1D != 0);
            auto const result(alpaka::block::sync::syncBlockThreadsPredicate<alpaka::block::sync::op::LogicalAnd>(acc, predicate));
            ALPAKA_CHECK(*success, result == 0);
        }

        // syncBlockThreadsPredicate<alpaka::block::sync::op::LogicalOr>
        {
            int const predicate(1);
            auto const result(alpaka::block::sync::syncBlockThreadsPredicate<alpaka::block::sync::op::LogicalOr>(acc, predicate));
            ALPAKA_CHECK(*success, result == 1);
        }
        {
            int const predicate(0);
            auto const result(alpaka::block::sync::syncBlockThreadsPredicate<alpaka::block::sync::op::LogicalOr>(acc, predicate));
            ALPAKA_CHECK(*success, result == 0);
        }
        {
            int const predicate(static_cast<int>(blockThreadIdx1D != 1));
            auto const result(alpaka::block::sync::syncBlockThreadsPredicate<alpaka::block::sync::op::LogicalOr>(acc, predicate));
            ALPAKA_CHECK(*success, result == 1);
        }
    }
};

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( "synchronizePredicate", "[blockSync]", alpaka::test::acc::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::dim::Dim<Acc>;
    using Idx = alpaka::idx::Idx<Acc>;

    BlockSyncPredicateTestKernel kernel;

    // 4^Dim
    {
        alpaka::test::KernelExecutionFixture<Acc> fixture(
            alpaka::vec::Vec<Dim, Idx>::all(static_cast<Idx>(4u)));

        REQUIRE(
            fixture(
                kernel));
    }

    // 1^Dim
    {
        alpaka::test::KernelExecutionFixture<Acc> fixture(
            alpaka::vec::Vec<Dim, Idx>::ones());

        REQUIRE(
            fixture(
                kernel));
    }
}
