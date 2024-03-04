/* Copyright 2022 Sergei Bastrakov, Bernhard Manfred Gruber, Jan Stephan, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/warp/Traits.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>

template<std::uint32_t TWarpSize>
struct GetSizeTestKernel
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success, std::int32_t expectedWarpSize) const -> void
    {
        ALPAKA_CHECK(*success, alpaka::warp::getSize(acc) == expectedWarpSize);
    }
};

template<std::uint32_t TWarpSize, typename TAcc>
struct alpaka::trait::WarpSize<GetSizeTestKernel<TWarpSize>, TAcc> : std::integral_constant<std::uint32_t, TWarpSize>
{
};

TEMPLATE_LIST_TEST_CASE("getSize", "[warp]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    auto const platform = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platform, 0);
    auto const warpSizes = alpaka::getWarpSizes(dev);
    REQUIRE(std::any_of(
        begin(warpSizes),
        end(warpSizes),
        [](std::size_t ws)
        {
            alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::all(8));
            if(ws == 4)
            {
                return fixture(GetSizeTestKernel<4>{}, static_cast<std::int32_t>(ws));
            }
            else if(ws == 8)
            {
                return fixture(GetSizeTestKernel<8>{}, static_cast<std::int32_t>(ws));
            }
            else if(ws == 16)
            {
                return fixture(GetSizeTestKernel<16>{}, static_cast<std::int32_t>(ws));
            }
            else if(ws == 32)
            {
                return fixture(GetSizeTestKernel<32>{}, static_cast<std::int32_t>(ws));
            }
            else if(ws == 64)
            {
                return fixture(GetSizeTestKernel<64>{}, static_cast<std::int32_t>(ws));
            }
            return fixture(GetSizeTestKernel<0>{}, static_cast<std::int32_t>(ws));
        }));
}
