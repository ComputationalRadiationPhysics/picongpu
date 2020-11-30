/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/meta/Transform.hpp>

#include <catch2/catch.hpp>

#include <tuple>
#include <type_traits>

template<typename T>
using AddConst = T const;

//-----------------------------------------------------------------------------
TEST_CASE("transform", "[meta]")
{
    using TransformInput = std::tuple<int, float, long>;

    using TransformResult = alpaka::meta::Transform<TransformInput, AddConst>;

    using TransformReference = std::tuple<int const, float const, long const>;

    static_assert(std::is_same<TransformReference, TransformResult>::value, "alpaka::meta::Transform failed!");
}

//-----------------------------------------------------------------------------
TEST_CASE("transformVariadic", "[meta]")
{
    using TransformInput = std::tuple<int, float, long>;

    using TransformResult = alpaka::meta::Transform<TransformInput, std::tuple>;

    using TransformReference = std::tuple<std::tuple<int>, std::tuple<float>, std::tuple<long>>;

    static_assert(std::is_same<TransformReference, TransformResult>::value, "alpaka::meta::Transform failed!");
}
