/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/meta/Apply.hpp>

#include <catch2/catch_test_macros.hpp>

#include <tuple>
#include <type_traits>

template<typename... T>
struct TypeList
{
};

TEST_CASE("apply", "[meta]")
{
    using ApplyInput = std::tuple<int, float, long>;

    using ApplyResult = alpaka::meta::Apply<ApplyInput, TypeList>;

    using ApplyReference = TypeList<int, float, long>;

    static_assert(std::is_same_v<ApplyReference, ApplyResult>, "alpaka::meta::Apply failed!");
}
