/* Copyright 2022 Jiří Vyskočil, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/meta/IsArrayOrVector.hpp>
#include <alpaka/vec/Vec.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <string>
#include <vector>

TEST_CASE("isArrayOrVector", "[meta]")
{
    STATIC_REQUIRE(alpaka::meta::IsArrayOrVector<std::array<int, 10>>::value);
    STATIC_REQUIRE(alpaka::meta::IsArrayOrVector<std::vector<float>>::value);
    STATIC_REQUIRE(alpaka::meta::IsArrayOrVector<alpaka::Vec<alpaka::DimInt<6u>, float>>::value);

    [[maybe_unused]] float arrayFloat[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    STATIC_REQUIRE(alpaka::meta::IsArrayOrVector<decltype(arrayFloat)>::value);
}

TEST_CASE("isActuallyNotArrayOrVector", "[meta]")
{
    float notAnArrayFloat = 15.0f;
    STATIC_REQUIRE_FALSE(alpaka::meta::IsArrayOrVector<decltype(notAnArrayFloat)>::value);

    [[maybe_unused]] float* notAnArrayFloatPointer = &notAnArrayFloat;
    STATIC_REQUIRE_FALSE(alpaka::meta::IsArrayOrVector<decltype(notAnArrayFloatPointer)>::value);

    std::string notAnArrayString{"alpaka"};
    STATIC_REQUIRE_FALSE(alpaka::meta::IsArrayOrVector<decltype(notAnArrayString)>::value);
}
