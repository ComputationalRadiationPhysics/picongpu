/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Jan Stephan
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/meta/Set.hpp>

#include <catch2/catch_test_macros.hpp>

#include <tuple>
#include <type_traits>

TEST_CASE("isSetTrue", "[meta]")
{
    using IsSetInput = std::tuple<int, float, long>;

    constexpr bool IsSetResult = alpaka::meta::IsSet<IsSetInput>::value;

    constexpr bool IsSetReference = true;

    static_assert(IsSetReference == IsSetResult, "alpaka::meta::IsSet failed!");
}

TEST_CASE("isSetFalse", "[meta]")
{
    using IsSetInput = std::tuple<int, float, int>;

    constexpr bool IsSetResult = alpaka::meta::IsSet<IsSetInput>::value;

    constexpr bool IsSetReference = false;

    static_assert(IsSetReference == IsSetResult, "alpaka::meta::IsSet failed!");
}
