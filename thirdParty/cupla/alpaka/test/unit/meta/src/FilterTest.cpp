/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/meta/Filter.hpp>

#include <catch2/catch.hpp>

#include <tuple>
#include <type_traits>

//-----------------------------------------------------------------------------
TEST_CASE("filter", "[meta]")
{
    using FilterInput = std::tuple<int, float, long>;

    using FilterResult = alpaka::meta::Filter<FilterInput, std::is_integral>;

    using FilterReference = std::tuple<int, long>;

    static_assert(std::is_same<FilterReference, FilterResult>::value, "alpaka::meta::Filter failed!");
}
