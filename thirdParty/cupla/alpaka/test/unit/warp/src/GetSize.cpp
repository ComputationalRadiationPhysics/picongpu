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
class GetSizeTestKernel
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        bool * success,
        std::int32_t expectedWarpSize) const
    -> void
    {
        std::int32_t const actualWarpSize = alpaka::warp::getSize(acc);
        ALPAKA_CHECK(*success, actualWarpSize == expectedWarpSize);
    }
};

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( "getSize", "[warp]", alpaka::test::acc::TestAccs)
{
    using Acc = TestType;
    using Dev = alpaka::dev::Dev<Acc>;
    using Pltf = alpaka::pltf::Pltf<Dev>;
    using Dim = alpaka::dim::Dim<Acc>;
    using Idx = alpaka::idx::Idx<Acc>;

    Dev const dev(alpaka::pltf::getDevByIdx<Pltf>(0u));
    auto const expectedWarpSize = static_cast<int>(alpaka::dev::getWarpSize(dev));
    Idx const gridThreadExtentPerDim = 8;
    alpaka::test::KernelExecutionFixture<Acc> fixture(
        alpaka::vec::Vec<Dim, Idx>::all(gridThreadExtentPerDim));
    GetSizeTestKernel kernel;
    REQUIRE(
        fixture(
            kernel,
            expectedWarpSize));
}
