/* Copyright 2022 Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/core/Utility.hpp>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("divCeil", "[core]")
{
    STATIC_REQUIRE(alpaka::core::divCeil(10, 2) == 5);
    STATIC_REQUIRE(alpaka::core::divCeil(11, 2) == 6);
    STATIC_REQUIRE(alpaka::core::divCeil(10, 3) == 4);
}

TEST_CASE("intPow", "[core]")
{
    STATIC_REQUIRE(alpaka::core::intPow(2, 0) == 1);
    STATIC_REQUIRE(alpaka::core::intPow(2, 1) == 2);
    STATIC_REQUIRE(alpaka::core::intPow(2, 4) == 16);
    STATIC_REQUIRE(alpaka::core::intPow(2, 10) == 1024);
}

TEST_CASE("nthRootFloor", "[core]")
{
    STATIC_REQUIRE(alpaka::core::nthRootFloor(8, 3) == 2);
    STATIC_REQUIRE(alpaka::core::nthRootFloor(1024, 3) == 10);
    STATIC_REQUIRE(alpaka::core::nthRootFloor(1000, 3) == 10);
    STATIC_REQUIRE(alpaka::core::nthRootFloor(1024, 2) == 32);
    STATIC_REQUIRE(alpaka::core::nthRootFloor(1024, 1) == 1024);
}
