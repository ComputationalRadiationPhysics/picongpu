/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/meta/Concatenate.hpp>

#include <catch2/catch.hpp>

#include <string>
#include <tuple>
#include <type_traits>

//-----------------------------------------------------------------------------
TEST_CASE("concatenate", "[meta]")
{
    using TestTuple1 = std::tuple<float, int, std::tuple<double, unsigned long>>;

    using TestTuple2 = std::tuple<bool, std::string>;

    using ConcatenateResult = alpaka::meta::Concatenate<TestTuple1, TestTuple2>;

    using ConcatenateReference = std::tuple<float, int, std::tuple<double, unsigned long>, bool, std::string>;

    static_assert(std::is_same<ConcatenateReference, ConcatenateResult>::value, "alpaka::meta::Concatenate failed!");
}
