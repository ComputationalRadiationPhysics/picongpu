/* Copyright 2023 Sergei Bastrakov, Bernhard Manfred Gruber, Jan Stephan, Andrea Bocci, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/acc/Tag.hpp>
#include <alpaka/meta/TypeListOps.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/warp/Traits.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <climits>
#include <cstdint>

struct ActivemaskSingleThreadWarpTestKernel
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        if constexpr(alpaka::Dim<TAcc>::value > 0)
            ALPAKA_CHECK(*success, alpaka::warp::getSize(acc) == 1);
        ALPAKA_CHECK(*success, alpaka::warp::activemask(acc) == 1u);
    }
};

template<std::uint32_t TWarpSize>
struct ActivemaskMultipleThreadWarpTestKernel
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success, std::uint64_t inactiveThreadIdx) const -> void
    {
        std::int32_t const warpExtent = alpaka::warp::getSize(acc);
        ALPAKA_CHECK(*success, warpExtent > 1);

        // Test relies on having a single warp per thread block
        auto const blockExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        ALPAKA_CHECK(*success, static_cast<std::int32_t>(blockExtent.prod()) == warpExtent);
        auto const localThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        auto const threadIdxInWarp = static_cast<std::uint64_t>(alpaka::mapIdx<1u>(localThreadIdx, blockExtent)[0]);

        if(threadIdxInWarp == inactiveThreadIdx)
            return;

        auto const actual = alpaka::warp::activemask(acc);
        using Result = decltype(actual);
        Result const allActive = static_cast<size_t>(warpExtent) == sizeof(Result) * CHAR_BIT
                                     ? ~Result{0u}
                                     : (Result{1} << warpExtent) - 1u;
        Result const expected = allActive & ~(Result{1} << inactiveThreadIdx);
        ALPAKA_CHECK(*success, actual == expected);
    }
};

template<std::uint32_t TWarpSize, typename TAcc>
struct alpaka::trait::WarpSize<ActivemaskMultipleThreadWarpTestKernel<TWarpSize>, TAcc>
    : std::integral_constant<std::uint32_t, TWarpSize>
{
};

TEMPLATE_LIST_TEST_CASE("activemask", "[warp]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    if constexpr(alpaka::accMatchesTags<
                     Acc,
                     alpaka::TagCpuSycl,
                     alpaka::TagGpuSyclIntel,
                     alpaka::TagFpgaSyclIntel,
                     alpaka::TagGenericSycl>)
    {
        std::cout << "Test disabled for SYCL\n";
        return;
    }
    else
    {
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
                CHECK(fixture(ActivemaskSingleThreadWarpTestKernel{}));
            }
            else
            {
                using ExecutionFixture = alpaka::test::KernelExecutionFixture<Acc>;
                auto const gridBlockExtent = alpaka::Vec<Dim, Idx>::all(2);
                // Enforce one warp per thread block
                auto blockThreadExtent = alpaka::Vec<Dim, Idx>::ones();
                blockThreadExtent[0] = static_cast<Idx>(warpExtent);
                auto const threadElementExtent = alpaka::Vec<Dim, Idx>::ones();
                auto workDiv =
                    typename ExecutionFixture::WorkDiv{gridBlockExtent, blockThreadExtent, threadElementExtent};
                auto fixture = ExecutionFixture{workDiv};
                if(warpExtent == 4)
                {
                    for(auto inactiveThreadIdx = 0u; inactiveThreadIdx < warpExtent; inactiveThreadIdx++)
                    {
                        CHECK(fixture(ActivemaskMultipleThreadWarpTestKernel<4>{}, inactiveThreadIdx));
                    }
                }
                else if(warpExtent == 8)
                {
                    for(auto inactiveThreadIdx = 0u; inactiveThreadIdx < warpExtent; inactiveThreadIdx++)
                    {
                        CHECK(fixture(ActivemaskMultipleThreadWarpTestKernel<8>{}, inactiveThreadIdx));
                    }
                }
                else if(warpExtent == 16)
                {
                    for(auto inactiveThreadIdx = 0u; inactiveThreadIdx < warpExtent; inactiveThreadIdx++)
                    {
                        CHECK(fixture(ActivemaskMultipleThreadWarpTestKernel<16>{}, inactiveThreadIdx));
                    }
                }
                else if(warpExtent == 32)
                {
                    for(auto inactiveThreadIdx = 0u; inactiveThreadIdx < warpExtent; inactiveThreadIdx++)
                    {
                        CHECK(fixture(ActivemaskMultipleThreadWarpTestKernel<32>{}, inactiveThreadIdx));
                    }
                }
                else if(warpExtent == 64)
                {
                    for(auto inactiveThreadIdx = 0u; inactiveThreadIdx < warpExtent; inactiveThreadIdx++)
                    {
                        CHECK(fixture(ActivemaskMultipleThreadWarpTestKernel<64>{}, inactiveThreadIdx));
                    }
                }
            }
        }
    }
}
