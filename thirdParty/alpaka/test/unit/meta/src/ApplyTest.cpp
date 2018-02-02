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
    typename... T>
struct TypeList
{};

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(apply)
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

BOOST_AUTO_TEST_SUITE_END()
