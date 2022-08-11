/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Bernhard Manfred Gruber, Jan Stephan
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/meta/CartesianProduct.hpp>

#include <catch2/catch_test_macros.hpp>

#include <tuple>
#include <type_traits>

TEST_CASE("cartesianProduct", "[meta]")
{
    using TestDims = std::tuple<alpaka::DimInt<1u>, alpaka::DimInt<2u>, alpaka::DimInt<3u>>;

    using TestIdxs = std::tuple<std::size_t, std::int64_t>;

    using CartesianProductResult = alpaka::meta::CartesianProduct<std::tuple, TestDims, TestIdxs>;

    using CartesianProductReference = std::tuple<
        std::tuple<alpaka::DimInt<1u>, std::size_t>,
        std::tuple<alpaka::DimInt<2u>, std::size_t>,
        std::tuple<alpaka::DimInt<3u>, std::size_t>,
        std::tuple<alpaka::DimInt<1u>, std::int64_t>,
        std::tuple<alpaka::DimInt<2u>, std::int64_t>,
        std::tuple<alpaka::DimInt<3u>, std::int64_t>>;

    static_assert(
        std::is_same_v<CartesianProductReference, CartesianProductResult>,
        "alpaka::meta::CartesianProduct failed!");
}
