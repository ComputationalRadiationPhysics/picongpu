/* Copyright 2020 Axel Huebl, Benjamin Worpitz, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/block/sync/Traits.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

class BlockSyncPredicateTestKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        using Idx = alpaka::Idx<TAcc>;

        // Get the index of the current thread within the block and the block extent and map them to 1D.
        auto const blockThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        auto const blockThreadExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        auto const blockThreadIdx1D = alpaka::mapIdx<1u>(blockThreadIdx, blockThreadExtent)[0u];
        auto const blockThreadExtent1D = blockThreadExtent.prod();

        // syncBlockThreadsPredicate<alpaka::BlockCount>
        {
            Idx const modulus = 2u;
            int const predicate = static_cast<int>(blockThreadIdx1D % modulus);
            auto const result = alpaka::syncBlockThreadsPredicate<alpaka::BlockCount>(acc, predicate);
            auto const expectedResult = static_cast<int>(blockThreadExtent1D / modulus);
            ALPAKA_CHECK(*success, expectedResult == result);
        }
        {
            Idx const modulus = 3u;
            int const predicate = static_cast<int>(blockThreadIdx1D % modulus);
            auto const result = alpaka::syncBlockThreadsPredicate<alpaka::BlockCount>(acc, predicate);
            auto const expectedResult = static_cast<int>(
                blockThreadExtent1D - ((blockThreadExtent1D + modulus - static_cast<Idx>(1u)) / modulus));
            ALPAKA_CHECK(*success, expectedResult == result);
        }

        // syncBlockThreadsPredicate<alpaka::BlockAnd>
        {
            int const predicate = 1;
            auto const result = alpaka::syncBlockThreadsPredicate<alpaka::BlockAnd>(acc, predicate);
            ALPAKA_CHECK(*success, result == 1);
        }
        {
            int const predicate = 0;
            auto const result = alpaka::syncBlockThreadsPredicate<alpaka::BlockAnd>(acc, predicate);
            ALPAKA_CHECK(*success, result == 0);
        }
        {
            int const predicate = blockThreadIdx1D != 0;
            auto const result = alpaka::syncBlockThreadsPredicate<alpaka::BlockAnd>(acc, predicate);
            ALPAKA_CHECK(*success, result == 0);
        }

        // syncBlockThreadsPredicate<alpaka::BlockOr>
        {
            int const predicate = 1;
            auto const result = alpaka::syncBlockThreadsPredicate<alpaka::BlockOr>(acc, predicate);
            ALPAKA_CHECK(*success, result == 1);
        }
        {
            int const predicate = 0;
            auto const result = alpaka::syncBlockThreadsPredicate<alpaka::BlockOr>(acc, predicate);
            ALPAKA_CHECK(*success, result == 0);
        }
        {
            int const predicate = static_cast<int>(blockThreadIdx1D != 1);
            auto const result = alpaka::syncBlockThreadsPredicate<alpaka::BlockOr>(acc, predicate);
            ALPAKA_CHECK(*success, result == 1);
        }
    }
};

TEMPLATE_LIST_TEST_CASE("synchronizePredicate", "[blockSync]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    BlockSyncPredicateTestKernel kernel;

    // 4^Dim
    {
        alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::all(static_cast<Idx>(4u)));

        REQUIRE(fixture(kernel));
    }

    // 1^Dim
    {
        alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

        REQUIRE(fixture(kernel));
    }
}
