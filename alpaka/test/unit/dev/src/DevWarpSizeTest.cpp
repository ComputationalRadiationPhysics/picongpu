/* Copyright 2024 Sergei Bastrakov, Bernhard Manfred Gruber, Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/dev/Traits.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>

TEMPLATE_LIST_TEST_CASE("getWarpSizes", "[dev]", alpaka::test::TestAccs)
{
    auto const platform = alpaka::Platform<TestType>{};
    auto const dev = alpaka::getDevByIdx(platform, 0);
    auto const warpExtents = alpaka::getWarpSizes(dev);
    REQUIRE(std::all_of(
        std::cbegin(warpExtents),
        std::cend(warpExtents),
        [](std::size_t warpExtent) { return warpExtent > 0; }));
}

TEMPLATE_LIST_TEST_CASE("getPreferredWarpSize", "[dev]", alpaka::test::TestAccs)
{
    auto const platform = alpaka::Platform<TestType>{};
    auto const dev = alpaka::getDevByIdx(platform, 0);
    auto const preferredWarpSize = alpaka::getPreferredWarpSize(dev);
    REQUIRE(preferredWarpSize > 0);
}

TEMPLATE_LIST_TEST_CASE("isDevice", "[dev]", alpaka::test::TestAccs)
{
    auto const platform = alpaka::Platform<TestType>{};
    auto const dev = alpaka::getDevByIdx(platform, 0);
    REQUIRE(alpaka::isDevice<decltype(dev)>);
}
