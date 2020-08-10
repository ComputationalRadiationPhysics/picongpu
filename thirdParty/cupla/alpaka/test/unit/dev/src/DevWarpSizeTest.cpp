/* Copyright 2020 Sergei Bastrakov
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/dev/Traits.hpp>

#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch.hpp>

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( "getWarpSize", "[dev]", alpaka::test::acc::TestAccs)
{
    using Dev = alpaka::dev::Dev<TestType>;
    using Pltf = alpaka::pltf::Pltf<Dev>;
    Dev const dev(alpaka::pltf::getDevByIdx<Pltf>(0u));
    auto const warpExtent = alpaka::dev::getWarpSize(dev);
    REQUIRE(warpExtent > 0);
}
