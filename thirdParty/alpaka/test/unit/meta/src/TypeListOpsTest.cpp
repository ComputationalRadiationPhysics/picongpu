/* Copyright 2022 Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/meta/TypeListOps.hpp>

#include <catch2/catch_test_macros.hpp>

#include <tuple>
#include <type_traits>

TEST_CASE("front", "[meta]")
{
    STATIC_REQUIRE(std::is_same_v<alpaka::meta::Front<std::tuple<int>>, int>);
    STATIC_REQUIRE(std::is_same_v<alpaka::meta::Front<std::tuple<int, int>>, int>);
    STATIC_REQUIRE(std::is_same_v<alpaka::meta::Front<std::tuple<float, int>>, float>);
    STATIC_REQUIRE(std::is_same_v<alpaka::meta::Front<std::tuple<short, int, double, float, float>>, short>);
}

TEST_CASE("contains", "[meta]")
{
    STATIC_REQUIRE(!alpaka::meta::Contains<std::tuple<>, int>::value);
    STATIC_REQUIRE(alpaka::meta::Contains<std::tuple<int>, int>::value);
    STATIC_REQUIRE(alpaka::meta::Contains<std::tuple<short, int, double, float>, short>::value);
    STATIC_REQUIRE(alpaka::meta::Contains<std::tuple<short, int, double, float>, double>::value);
    STATIC_REQUIRE(alpaka::meta::Contains<std::tuple<short, int, double, float>, float>::value);
    STATIC_REQUIRE(!alpaka::meta::Contains<std::tuple<short, int, double, float>, char>::value);
}
