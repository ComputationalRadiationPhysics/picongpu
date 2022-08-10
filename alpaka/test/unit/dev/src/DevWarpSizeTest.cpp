/* Copyright 2022 Sergei Bastrakov, Bernhard Manfred Gruber, Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/dev/Traits.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>

TEMPLATE_LIST_TEST_CASE("getWarpSizes", "[dev]", alpaka::test::TestAccs)
{
    using Dev = alpaka::Dev<TestType>;
    using Pltf = alpaka::Pltf<Dev>;
    Dev const dev(alpaka::getDevByIdx<Pltf>(0u));
    auto const warpExtents = alpaka::getWarpSizes(dev);
    REQUIRE(std::all_of(
        std::cbegin(warpExtents),
        std::cend(warpExtents),
        [](std::size_t warpExtent) { return warpExtent > 0; }));
}
