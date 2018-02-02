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
BOOST_AUTO_TEST_CASE(isSetTrue)
{
    using IsSetInput =
        std::tuple<
            int,
            float,
            long>;

    constexpr bool IsSetResult =
        alpaka::meta::IsSet<
            IsSetInput
        >::value;

    constexpr bool IsSetReference =
        true;

    static_assert(
        IsSetReference == IsSetResult,
        "alpaka::meta::IsSet failed!");
}

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(isSetFalse)
{
    using IsSetInput =
        std::tuple<
            int,
            float,
            int>;

    constexpr bool IsSetResult =
        alpaka::meta::IsSet<
            IsSetInput
        >::value;

    constexpr bool IsSetReference =
        false;

    static_assert(
        IsSetReference == IsSetResult,
        "alpaka::meta::IsSet failed!");
}

BOOST_AUTO_TEST_SUITE_END()
