/**
 * \file
 * Copyright 2015 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * alpaka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * alpaka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with alpaka.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <alpaka/alpaka.hpp>

#include <boost/predef.h>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include <boost/test/unit_test.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif

#include <tuple>
#include <type_traits>

BOOST_AUTO_TEST_SUITE(meta)

class A {};
class B : A {};
class C {};

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(isStrictBaseTrue)
{
    constexpr bool IsStrictBaseResult =
        alpaka::meta::IsStrictBase<
            A, B
        >::value;

    constexpr bool IsStrictBaseReference =
        true;

    static_assert(
        IsStrictBaseReference == IsStrictBaseResult,
        "alpaka::meta::IsStrictBase failed!");
}

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(isStrictBaseIdentity)
{
    constexpr bool IsStrictBaseResult =
        alpaka::meta::IsStrictBase<
            A, A
        >::value;

    constexpr bool IsStrictBaseReference =
        false;

    static_assert(
        IsStrictBaseReference == IsStrictBaseResult,
        "alpaka::meta::IsStrictBase failed!");
}

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(isStrictBaseNoInheritance)
{
    constexpr bool IsStrictBaseResult =
        alpaka::meta::IsStrictBase<
            A, C
        >::value;

    constexpr bool IsStrictBaseReference =
        false;

    static_assert(
        IsStrictBaseReference == IsStrictBaseResult,
        "alpaka::meta::IsStrictBase failed!");
}

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(isStrictBaseWrongOrder)
{
    constexpr bool IsStrictBaseResult =
        alpaka::meta::IsStrictBase<
            B, A
        >::value;

    constexpr bool IsStrictBaseReference =
        false;

    static_assert(
        IsStrictBaseReference == IsStrictBaseResult,
        "alpaka::meta::IsStrictBase failed!");
}

BOOST_AUTO_TEST_SUITE_END()
