/* Copyright 2022 Sergei Bastrakov, Bernhard Manfred Gruber, Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/warp/Traits.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <climits>
#include <cstdint>

struct BallotSingleThreadWarpTestKernel
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        if constexpr(alpaka::Dim<TAcc>::value > 0)
            ALPAKA_CHECK(*success, alpaka::warp::getSize(acc) == 1);
        ALPAKA_CHECK(*success, alpaka::warp::ballot(acc, 42) == 1u);
        ALPAKA_CHECK(*success, alpaka::warp::ballot(acc, 0) == 0u);
    }
};

struct BallotMultipleThreadWarpTestKernel
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        std::int32_t const warpExtent = alpaka::warp::getSize(acc);
        ALPAKA_CHECK(*success, warpExtent > 1);

        using BallotResultType = decltype(alpaka::warp::ballot(acc, 42));
        BallotResultType const allActive = static_cast<size_t>(warpExtent) == sizeof(BallotResultType) * CHAR_BIT
                                               ? ~BallotResultType{0u}
                                               : (BallotResultType{1} << warpExtent) - 1u;
        ALPAKA_CHECK(*success, alpaka::warp::ballot(acc, 42) == allActive);
        ALPAKA_CHECK(*success, alpaka::warp::ballot(acc, 0) == 0u);

        // Test relies on having a single warp per thread block
        auto const blockExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        ALPAKA_CHECK(*success, static_cast<std::int32_t>(blockExtent.prod()) == warpExtent);
        auto const localThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        auto const threadIdxInWarp = static_cast<std::int32_t>(alpaka::mapIdx<1u>(localThreadIdx, blockExtent)[0]);

        // Some threads quit the kernel to test that the warp operations
        // properly operate on the active threads only
        if(threadIdxInWarp >= warpExtent / 2)
            return;

        for(auto idx = 0; idx < warpExtent / 2; idx++)
        {
            ALPAKA_CHECK(
                *success,
                alpaka::warp::ballot(acc, threadIdxInWarp == idx ? 1 : 0) == std::uint64_t{1} << idx);
            // First warpExtent / 2 bits are 1 except bit idx
            std::uint64_t const expected = ((std::uint64_t{1} << warpExtent / 2) - 1) & ~(std::uint64_t{1} << idx);
            ALPAKA_CHECK(*success, alpaka::warp::ballot(acc, threadIdxInWarp == idx ? 0 : 1) == expected);
        }
    }
};

TEMPLATE_LIST_TEST_CASE("ballot", "[warp]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dev = alpaka::Dev<Acc>;
    using Pltf = alpaka::Pltf<Dev>;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    Dev const dev(alpaka::getDevByIdx<Pltf>(0u));
    auto const warpExtents = alpaka::getWarpSizes(dev);
    for(auto const warpExtent : warpExtents)
    {
        auto const scalar = Dim::value == 0 || warpExtent == 1;
        if(scalar)
        {
            alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::all(4));
            REQUIRE(fixture(BallotSingleThreadWarpTestKernel{}));
        }
        else
        {
            // Work around gcc 7.5 trying and failing to offload for OpenMP 4.0
#if BOOST_COMP_GNUC && (BOOST_COMP_GNUC == BOOST_VERSION_NUMBER(7, 5, 0)) && defined ALPAKA_ACC_ANY_BT_OMP5_ENABLED
            return;
#else
            using ExecutionFixture = alpaka::test::KernelExecutionFixture<Acc>;
            auto const gridBlockExtent = alpaka::Vec<Dim, Idx>::all(2);
            // Enforce one warp per thread block
            auto blockThreadExtent = alpaka::Vec<Dim, Idx>::ones();
            blockThreadExtent[0] = static_cast<Idx>(warpExtent);
            auto const threadElementExtent = alpaka::Vec<Dim, Idx>::ones();
            auto workDiv = typename ExecutionFixture::WorkDiv{gridBlockExtent, blockThreadExtent, threadElementExtent};
            auto fixture = ExecutionFixture{workDiv};
            BallotMultipleThreadWarpTestKernel kernel;
            REQUIRE(fixture(kernel));
#endif
        }
    }
}
