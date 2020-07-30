/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/meta/Apply.hpp>

#include <catch2/catch.hpp>

#include <tuple>
#include <type_traits>

template<
    typename... T>
struct TypeList
{};

//-----------------------------------------------------------------------------
TEST_CASE("apply", "[meta]")
{
    using ApplyInput =
        std::tuple<
            int,
            float,
            long>;

    using ApplyResult =
        alpaka::meta::Apply<
            ApplyInput,
            TypeList
        >;

    using ApplyReference =
        TypeList<
            int,
            float,
            long>;

    static_assert(
        std::is_same<
            ApplyReference,
            ApplyResult
        >::value,
        "alpaka::meta::Apply failed!");
}
