/* Copyright 2020 Sergei Bastrakov
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/warp/Traits.hpp>

#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>

#include <catch2/catch.hpp>

#include <cstdint>

//#############################################################################
class AllSingleThreadWarpTestKernel
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
        std::int32_t const warpExtent = alpaka::warp::getSize(acc);
        ALPAKA_CHECK(*success, warpExtent == 1);

        ALPAKA_CHECK(*success, alpaka::warp::all(acc, 42) != 0);
        ALPAKA_CHECK(*success, alpaka::warp::all(acc, 0) == 0);
    }
};

//#############################################################################
class AllMultipleThreadWarpTestKernel
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
        std::int32_t const warpExtent = alpaka::warp::getSize(acc);
        ALPAKA_CHECK(*success, warpExtent > 1);

        ALPAKA_CHECK(*success, alpaka::warp::all(acc, 0) == 0);
        ALPAKA_CHECK(*success, alpaka::warp::all(acc, 42) != 0);

        // Test relies on having a single warp per thread block
        auto const blockExtent = alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        ALPAKA_CHECK(*success, static_cast<std::int32_t>(blockExtent.prod()) == warpExtent);
        auto const localThreadIdx = alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc);
        auto const threadIdxInWarp = static_cast<std::int32_t>(alpaka::idx::mapIdx<1u>(
            localThreadIdx,
            blockExtent)[0]);

        // Some threads quit the kernel to test that the warp operations
        // properly operate on the active threads only
        if (threadIdxInWarp % 3)
            return;

        for (auto idx = 0; idx < warpExtent; idx++)
        {
            ALPAKA_CHECK(
                *success,
                alpaka::warp::all(acc, threadIdxInWarp == idx ? 1 : 0) == 0);
            std::int32_t const expected = idx % 3 ? 1 : 0;
            ALPAKA_CHECK(
                *success,
                alpaka::warp::all(acc, threadIdxInWarp == idx ? 0 : 1) == expected);
        }
    }
};

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( "all", "[warp]", alpaka::test::acc::TestAccs)
{
    using Acc = TestType;
    using Dev = alpaka::dev::Dev<Acc>;
    using Pltf = alpaka::pltf::Pltf<Dev>;
    using Dim = alpaka::dim::Dim<Acc>;
    using Idx = alpaka::idx::Idx<Acc>;

    Dev const dev(alpaka::pltf::getDevByIdx<Pltf>(0u));
    auto const warpExtent = alpaka::dev::getWarpSize(dev);
    if (warpExtent == 1)
    {
        Idx const gridThreadExtentPerDim = 4;
        alpaka::test::KernelExecutionFixture<Acc> fixture(
            alpaka::vec::Vec<Dim, Idx>::all(gridThreadExtentPerDim));
        AllSingleThreadWarpTestKernel kernel;
        REQUIRE(
            fixture(
                kernel));
    }
    else
    {
        // Work around gcc 7.5 trying and failing to offload for OpenMP 4.0
#if BOOST_COMP_GNUC && (BOOST_COMP_GNUC == BOOST_VERSION_NUMBER(7, 5, 0)) && ALPAKA_ACC_CPU_BT_OMP4_ENABLED
        return;
#else
        using ExecutionFixture = alpaka::test::KernelExecutionFixture<Acc>;
        auto const gridBlockExtent = alpaka::vec::Vec<Dim, Idx>::all(2);
        // Enforce one warp per thread block
        auto blockThreadExtent = alpaka::vec::Vec<Dim, Idx>::ones();
        blockThreadExtent[0] = static_cast<Idx>(warpExtent);
        auto const threadElementExtent = alpaka::vec::Vec<Dim, Idx>::ones();
        auto workDiv = typename ExecutionFixture::WorkDiv{
            gridBlockExtent,
            blockThreadExtent,
            threadElementExtent};
        auto fixture = ExecutionFixture{ workDiv };
        AllMultipleThreadWarpTestKernel kernel;
        REQUIRE(
            fixture(
                kernel));
#endif
    }
}
