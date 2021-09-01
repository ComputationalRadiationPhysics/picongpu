/* Copyright 2020 Sergei Bastrakov
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/meta/Void.hpp>

#include <catch2/catch.hpp>

#include <type_traits>
#include <vector>

//-----------------------------------------------------------------------------
TEST_CASE("voidNonEmpty", "[meta]")
{
    using Result = alpaka::meta::Void<int, float, int>;
    REQUIRE(std::is_same<void, Result>::value);
}

//-----------------------------------------------------------------------------
TEST_CASE("voidEmpty", "[meta]")
{
    using Result = alpaka::meta::Void<>;
    REQUIRE(std::is_same<void, Result>::value);
}

//-----------------------------------------------------------------------------
//#############################################################################
//! Trait to detect if the given class has a method size().
//! This illustrates and tests the technique of using Void<> to compile-time
//! check for methods (and members can be treated similarly).
template<class T, class = void>
struct HasMethodSize : std::false_type
{
};

template<class T>
struct HasMethodSize<T, alpaka::meta::Void<decltype(std::declval<T&>().size())>> : std::true_type
{
};

TEST_CASE("voidSFINAE", "[meta]")
{
    using DoesIntHaveMethodSize = HasMethodSize<int>;
    REQUIRE(false == DoesIntHaveMethodSize::value);

    using DoesVectorHaveMethodSize = HasMethodSize<std::vector<float>>;
    REQUIRE(true == DoesVectorHaveMethodSize::value);
}
