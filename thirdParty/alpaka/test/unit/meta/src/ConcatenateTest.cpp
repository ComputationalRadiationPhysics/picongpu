/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/meta/Concatenate.hpp>

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <tuple>
#include <type_traits>

TEST_CASE("concatenate", "[meta]")
{
    using TestTuple1 = std::tuple<float, int, std::tuple<double, unsigned long>>;

    using TestTuple2 = std::tuple<bool, std::string>;

    using ConcatenateResult = alpaka::meta::Concatenate<TestTuple1, TestTuple2>;

    using ConcatenateReference = std::tuple<float, int, std::tuple<double, unsigned long>, bool, std::string>;

    static_assert(std::is_same_v<ConcatenateReference, ConcatenateResult>, "alpaka::meta::Concatenate failed!");
}
