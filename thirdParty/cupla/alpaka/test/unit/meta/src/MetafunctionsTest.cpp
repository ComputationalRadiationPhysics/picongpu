/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/meta/Metafunctions.hpp>

#include <catch2/catch.hpp>

#include <tuple>
#include <type_traits>

//-----------------------------------------------------------------------------
TEST_CASE("conjunctionTrue", "[meta]")
{
    using ConjunctionResult
        = alpaka::meta::Conjunction<std::true_type, std::true_type, std::integral_constant<bool, true>>;

    static_assert(ConjunctionResult::value == true, "alpaka::meta::Conjunction failed!");
}

//-----------------------------------------------------------------------------
TEST_CASE("conjunctionFalse", "[meta]")
{
    using ConjunctionResult
        = alpaka::meta::Conjunction<std::true_type, std::false_type, std::integral_constant<bool, true>>;

    static_assert(ConjunctionResult::value == false, "alpaka::meta::Conjunction failed!");
}

//-----------------------------------------------------------------------------
TEST_CASE("disjunctionTrue", "[meta]")
{
    using DisjunctionResult
        = alpaka::meta::Disjunction<std::false_type, std::true_type, std::integral_constant<bool, false>>;

    static_assert(DisjunctionResult::value == true, "alpaka::meta::Disjunction failed!");
}

//-----------------------------------------------------------------------------
TEST_CASE("disjunctionFalse", "[meta]")
{
    using DisjunctionResult
        = alpaka::meta::Disjunction<std::false_type, std::false_type, std::integral_constant<bool, false>>;

    static_assert(DisjunctionResult::value == false, "alpaka::meta::Disjunction failed!");
}

//-----------------------------------------------------------------------------
TEST_CASE("negationFalse", "[meta]")
{
    using NegationResult = alpaka::meta::Negation<std::true_type>;

    using NegationReference = std::false_type;

    static_assert(std::is_same<NegationReference, NegationResult>::value, "alpaka::meta::Negation failed!");
}

//-----------------------------------------------------------------------------
TEST_CASE("negationTrue", "[meta]")
{
    using NegationResult = alpaka::meta::Negation<std::false_type>;

    using NegationReference = std::true_type;

    static_assert(std::is_same<NegationReference, NegationResult>::value, "alpaka::meta::Negation failed!");
}
