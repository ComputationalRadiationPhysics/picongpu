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

template<
    typename T>
using AddConst = T const;

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(transform)
{
    using TransformInput =
        std::tuple<
            int,
            float,
            long>;

    using TransformResult =
        alpaka::meta::Transform<
            TransformInput,
            AddConst
        >;

    using TransformReference =
        std::tuple<
            int const,
            float const,
            long const>;

    static_assert(
        std::is_same<
            TransformReference,
            TransformResult
        >::value,
        "alpaka::meta::Transform failed!");
}

BOOST_AUTO_TEST_SUITE_END()
