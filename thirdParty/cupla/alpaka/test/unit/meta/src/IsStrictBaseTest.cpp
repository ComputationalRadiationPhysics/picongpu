/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/meta/IsStrictBase.hpp>

#include <catch2/catch_test_macros.hpp>

#include <tuple>
#include <type_traits>

class A
{
};

class B : A
{
};

class C
{
};

TEST_CASE("isStrictBaseTrue", "[meta]")
{
    constexpr bool IsStrictBaseResult = alpaka::meta::IsStrictBase<A, B>::value;

    constexpr bool IsStrictBaseReference = true;

    static_assert(IsStrictBaseReference == IsStrictBaseResult, "alpaka::meta::IsStrictBase failed!");
}

TEST_CASE("isStrictBaseIdentity", "[meta]")
{
    constexpr bool IsStrictBaseResult = alpaka::meta::IsStrictBase<A, A>::value;

    constexpr bool IsStrictBaseReference = false;

    static_assert(IsStrictBaseReference == IsStrictBaseResult, "alpaka::meta::IsStrictBase failed!");
}

TEST_CASE("isStrictBaseNoInheritance", "[meta]")
{
    constexpr bool IsStrictBaseResult = alpaka::meta::IsStrictBase<A, C>::value;

    constexpr bool IsStrictBaseReference = false;

    static_assert(IsStrictBaseReference == IsStrictBaseResult, "alpaka::meta::IsStrictBase failed!");
}

TEST_CASE("isStrictBaseWrongOrder", "[meta]")
{
    constexpr bool IsStrictBaseResult = alpaka::meta::IsStrictBase<B, A>::value;

    constexpr bool IsStrictBaseReference = false;

    static_assert(IsStrictBaseReference == IsStrictBaseResult, "alpaka::meta::IsStrictBase failed!");
}
