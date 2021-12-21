/* Copyright 2020 Sergei Bastrakov
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

#include <catch2/catch.hpp>

#include <cstdint>

//#############################################################################
class AnySingleThreadWarpTestKernel
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        std::int32_t const warpExtent = alpaka::warp::getSize(acc);
        ALPAKA_CHECK(*success, warpExtent == 1);

        ALPAKA_CHECK(*success, alpaka::warp::any(acc, 42) != 0);
        ALPAKA_CHECK(*success, alpaka::warp::any(acc, 0) == 0);
    }
};

//#############################################################################
class AnyMultipleThreadWarpTestKernel
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        std::int32_t const warpExtent = alpaka::warp::getSize(acc);
        ALPAKA_CHECK(*success, warpExtent > 1);

        ALPAKA_CHECK(*success, alpaka::warp::any(acc, 0) == 0);
        ALPAKA_CHECK(*success, alpaka::warp::any(acc, 42) != 0);

        // Test relies on having a single warp per thread block
        auto const blockExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        ALPAKA_CHECK(*success, static_cast<std::int32_t>(blockExtent.prod()) == warpExtent);
        auto const localThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        auto const threadIdxInWarp = static_cast<std::int32_t>(alpaka::mapIdx<1u>(localThreadIdx, blockExtent)[0]);

        // Some threads quit the kernel to test that the warp operations
        // properly operate on the active threads only
        if(threadIdxInWarp % 5)
            return;

        for(auto idx = 0; idx < warpExtent; idx++)
        {
            ALPAKA_CHECK(*success, alpaka::warp::any(acc, threadIdxInWarp == idx ? 0 : 1) == 1);
            std::int32_t const expected = idx % 5 ? 0 : 1;
            ALPAKA_CHECK(*success, alpaka::warp::any(acc, threadIdxInWarp == idx ? 1 : 0) == expected);
        }
    }
};

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE("any", "[warp]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dev = alpaka::Dev<Acc>;
    using Pltf = alpaka::Pltf<Dev>;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    Dev const dev(alpaka::getDevByIdx<Pltf>(0u));
    auto const warpExtent = alpaka::getWarpSize(dev);
    if(warpExtent == 1)
    {
        Idx const gridThreadExtentPerDim = 4;
        alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::all(gridThreadExtentPerDim));
        AnySingleThreadWarpTestKernel kernel;
        REQUIRE(fixture(kernel));
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
        AnyMultipleThreadWarpTestKernel kernel;
        REQUIRE(fixture(kernel));
#endif
    }
}
