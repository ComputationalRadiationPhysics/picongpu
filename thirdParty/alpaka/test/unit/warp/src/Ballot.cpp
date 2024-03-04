/* Copyright 2023 Sergei Bastrakov, Bernhard Manfred Gruber, Jan Stephan, Andrea Bocci, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
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

template<std::uint32_t TWarpSize>
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

template<std::uint32_t TWarpSize, typename TAcc>
struct alpaka::trait::WarpSize<BallotMultipleThreadWarpTestKernel<TWarpSize>, TAcc>
    : std::integral_constant<std::uint32_t, TWarpSize>
{
};

TEMPLATE_LIST_TEST_CASE("ballot", "[warp]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    auto const platform = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platform, 0);
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
            using ExecutionFixture = alpaka::test::KernelExecutionFixture<Acc>;
            auto const gridBlockExtent = alpaka::Vec<Dim, Idx>::all(2);
            // Enforce one warp per thread block
            auto blockThreadExtent = alpaka::Vec<Dim, Idx>::ones();
            blockThreadExtent[0] = static_cast<Idx>(warpExtent);
            auto const threadElementExtent = alpaka::Vec<Dim, Idx>::ones();
            auto workDiv = typename ExecutionFixture::WorkDiv{gridBlockExtent, blockThreadExtent, threadElementExtent};
            auto fixture = ExecutionFixture{workDiv};
            if(warpExtent == 4)
            {
                REQUIRE(fixture(BallotMultipleThreadWarpTestKernel<4>{}));
            }
            else if(warpExtent == 8)
            {
                REQUIRE(fixture(BallotMultipleThreadWarpTestKernel<8>{}));
            }
            else if(warpExtent == 16)
            {
                REQUIRE(fixture(BallotMultipleThreadWarpTestKernel<16>{}));
            }
            else if(warpExtent == 32)
            {
                REQUIRE(fixture(BallotMultipleThreadWarpTestKernel<32>{}));
            }
            else if(warpExtent == 64)
            {
                REQUIRE(fixture(BallotMultipleThreadWarpTestKernel<64>{}));
            }
        }
    }
}
