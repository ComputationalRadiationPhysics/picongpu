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

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(cartesianProduct)
{
    using TestDims =
        std::tuple<
            alpaka::dim::DimInt<1u>,
            alpaka::dim::DimInt<2u>,
            alpaka::dim::DimInt<3u>>;

    using TestSizes =
        std::tuple<
            std::size_t,
            std::int64_t>;

    using CartesianProductResult =
        alpaka::meta::CartesianProduct<
            std::tuple,
            TestDims,
            TestSizes
        >;

    using CartesianProductReference =
        std::tuple<
            std::tuple<alpaka::dim::DimInt<1u>, std::size_t>,
            std::tuple<alpaka::dim::DimInt<2u>, std::size_t>,
            std::tuple<alpaka::dim::DimInt<3u>, std::size_t>,
            std::tuple<alpaka::dim::DimInt<1u>, std::int64_t>,
            std::tuple<alpaka::dim::DimInt<2u>, std::int64_t>,
            std::tuple<alpaka::dim::DimInt<3u>, std::int64_t>>;

    static_assert(
        std::is_same<
            CartesianProductReference,
            CartesianProductResult
        >::value,
        "alpaka::meta::CartesianProduct failed!");
}

BOOST_AUTO_TEST_SUITE_END()
