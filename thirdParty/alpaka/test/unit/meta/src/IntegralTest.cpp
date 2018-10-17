/**
 * \file
 * Copyright 2017 Benjamin Worpitz
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

#include <alpaka/core/BoostPredef.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include <boost/test/unit_test.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif

#include <type_traits>

BOOST_AUTO_TEST_SUITE(meta)

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(isIntegralSupersetTrue)
{
    // unsigned - unsigned
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint64_t, std::uint64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint64_t, std::uint32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint64_t, std::uint16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint64_t, std::uint8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint32_t, std::uint32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint32_t, std::uint16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint32_t, std::uint8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint16_t, std::uint16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint16_t, std::uint8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint8_t, std::uint8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    // signed - signed
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int64_t, std::int64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int64_t, std::int32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int64_t, std::int16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int64_t, std::int8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int32_t, std::int32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int32_t, std::int16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int32_t, std::int8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int16_t, std::int16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int16_t, std::int8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int8_t, std::int8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    // unsigned - signed
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint64_t, std::int32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint64_t, std::int16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint64_t, std::int8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint32_t, std::int16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint32_t, std::int8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint16_t, std::int8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    // signed - unsigned
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int64_t, std::uint32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int64_t, std::uint16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int64_t, std::uint8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int32_t, std::uint16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int32_t, std::uint8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int16_t, std::uint8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
}

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(isIntegralSupersetNoIntegral)
{
    static_assert(
        !alpaka::meta::IsIntegralSuperset<float, std::uint8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint64_t, double>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
}

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(isIntegralSupersetFalse)
{
    // unsigned - unsigned
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint8_t, std::uint64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint8_t, std::uint32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint8_t, std::uint16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint16_t, std::uint64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint16_t, std::uint32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint32_t, std::uint64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    // signed - signed
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int8_t, std::int64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int8_t, std::int32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int8_t, std::int16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int16_t, std::int64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int16_t, std::int32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int32_t, std::int64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    // unsigned - signed
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint64_t, std::int64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint32_t, std::int64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint32_t, std::int32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint16_t, std::int64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint16_t, std::int32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint16_t, std::int16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint8_t, std::int64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint8_t, std::int32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint8_t, std::int16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint8_t, std::int8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    // signed - unsigned
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int64_t, std::uint64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int32_t, std::uint64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int32_t, std::uint32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int16_t, std::uint64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int16_t, std::uint32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int16_t, std::uint16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int8_t, std::uint64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int8_t, std::uint32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int8_t, std::uint16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int8_t, std::uint8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
}

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(higherMax)
{
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int8_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int8_t, std::uint8_t>, std::uint8_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int8_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int8_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int8_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int8_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int8_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int8_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMax failed!");

    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint8_t, std::int8_t>, std::uint8_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint8_t, std::uint8_t>, std::uint8_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint8_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint8_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint8_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint8_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint8_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint8_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMax failed!");

    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int16_t, std::int8_t>, std::int16_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int16_t, std::uint8_t>, std::int16_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int16_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int16_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int16_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int16_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int16_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int16_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMax failed!");

    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint16_t, std::int8_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint16_t, std::uint8_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint16_t, std::int16_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint16_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint16_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint16_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint16_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint16_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMax failed!");

    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int32_t, std::int8_t>, std::int32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int32_t, std::uint8_t>, std::int32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int32_t, std::int16_t>, std::int32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int32_t, std::uint16_t>, std::int32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int32_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int32_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int32_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int32_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMax failed!");

    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint32_t, std::int8_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint32_t, std::uint8_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint32_t, std::int16_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint32_t, std::uint16_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint32_t, std::int32_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint32_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint32_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint32_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMax failed!");

    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int64_t, std::int8_t>, std::int64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int64_t, std::uint8_t>, std::int64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int64_t, std::int16_t>, std::int64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int64_t, std::uint16_t>, std::int64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int64_t, std::int32_t>, std::int64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int64_t, std::uint32_t>, std::int64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int64_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int64_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMax failed!");

    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint64_t, std::int8_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint64_t, std::uint8_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint64_t, std::int16_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint64_t, std::uint16_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint64_t, std::int32_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint64_t, std::uint32_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint64_t, std::int64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint64_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMax failed!");
}

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(lowerMax)
{
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int8_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int8_t, std::uint8_t>, std::int8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int8_t, std::int16_t>, std::int8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int8_t, std::uint16_t>, std::int8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int8_t, std::int32_t>, std::int8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int8_t, std::uint32_t>, std::int8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int8_t, std::int64_t>, std::int8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int8_t, std::uint64_t>, std::int8_t>::value,
        "alpaka::meta::LowerMax failed!");

    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint8_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint8_t, std::uint8_t>, std::uint8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint8_t, std::int16_t>, std::uint8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint8_t, std::uint16_t>, std::uint8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint8_t, std::int32_t>, std::uint8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint8_t, std::uint32_t>, std::uint8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint8_t, std::int64_t>, std::uint8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint8_t, std::uint64_t>, std::uint8_t>::value,
        "alpaka::meta::LowerMax failed!");

    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int16_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int16_t, std::uint8_t>, std::uint8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int16_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int16_t, std::uint16_t>, std::int16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int16_t, std::int32_t>, std::int16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int16_t, std::uint32_t>, std::int16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int16_t, std::int64_t>, std::int16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int16_t, std::uint64_t>, std::int16_t>::value,
        "alpaka::meta::LowerMax failed!");

    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint16_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint16_t, std::uint8_t>, std::uint8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint16_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint16_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint16_t, std::int32_t>, std::uint16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint16_t, std::uint32_t>, std::uint16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint16_t, std::int64_t>, std::uint16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint16_t, std::uint64_t>, std::uint16_t>::value,
        "alpaka::meta::LowerMax failed!");

    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int32_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int32_t, std::uint8_t>, std::uint8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int32_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int32_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int32_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int32_t, std::uint32_t>, std::int32_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int32_t, std::int64_t>, std::int32_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int32_t, std::uint64_t>, std::int32_t>::value,
        "alpaka::meta::LowerMax failed!");

    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint32_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint32_t, std::uint8_t>, std::uint8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint32_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint32_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint32_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint32_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint32_t, std::int64_t>, std::uint32_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint32_t, std::uint64_t>, std::uint32_t>::value,
        "alpaka::meta::LowerMax failed!");

    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int64_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int64_t, std::uint8_t>, std::uint8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int64_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int64_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int64_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int64_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int64_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int64_t, std::uint64_t>, std::int64_t>::value,
        "alpaka::meta::LowerMax failed!");

    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint64_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint64_t, std::uint8_t>, std::uint8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint64_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint64_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint64_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint64_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint64_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint64_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::LowerMax failed!");
}

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(higherMin)
{
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int8_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int8_t, std::uint8_t>, std::uint8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int8_t, std::int16_t>, std::int8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int8_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int8_t, std::int32_t>, std::int8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int8_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int8_t, std::int64_t>, std::int8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int8_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMin failed!");

    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint8_t, std::int8_t>, std::uint8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint8_t, std::uint8_t>, std::uint8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint8_t, std::int16_t>, std::uint8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint8_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint8_t, std::int32_t>, std::uint8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint8_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint8_t, std::int64_t>, std::uint8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint8_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMin failed!");

    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int16_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int16_t, std::uint8_t>, std::uint8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int16_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int16_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int16_t, std::int32_t>, std::int16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int16_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int16_t, std::int64_t>, std::int16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int16_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMin failed!");

    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint16_t, std::int8_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint16_t, std::uint8_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint16_t, std::int16_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint16_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint16_t, std::int32_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint16_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint16_t, std::int64_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint16_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMin failed!");

    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int32_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int32_t, std::uint8_t>, std::uint8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int32_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int32_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int32_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int32_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int32_t, std::int64_t>, std::int32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int32_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMin failed!");

    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint32_t, std::int8_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint32_t, std::uint8_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint32_t, std::int16_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint32_t, std::uint16_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint32_t, std::int32_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint32_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint32_t, std::int64_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint32_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMin failed!");

    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int64_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int64_t, std::uint8_t>, std::uint8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int64_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int64_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int64_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int64_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int64_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int64_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMin failed!");

    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint64_t, std::int8_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint64_t, std::uint8_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint64_t, std::int16_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint64_t, std::uint16_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint64_t, std::int32_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint64_t, std::uint32_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint64_t, std::int64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint64_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMin failed!");
}
//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(lowerMin)
{
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int8_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int8_t, std::uint8_t>, std::int8_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int8_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int8_t, std::uint16_t>, std::int8_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int8_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int8_t, std::uint32_t>, std::int8_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int8_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int8_t, std::uint64_t>, std::int8_t>::value,
        "alpaka::meta::LowerMin failed!");

    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint8_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint8_t, std::uint8_t>, std::uint8_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint8_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint8_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint8_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint8_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint8_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint8_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::LowerMin failed!");

    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int16_t, std::int8_t>, std::int16_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int16_t, std::uint8_t>, std::int16_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int16_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int16_t, std::uint16_t>, std::int16_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int16_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int16_t, std::uint32_t>, std::int16_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int16_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int16_t, std::uint64_t>, std::int16_t>::value,
        "alpaka::meta::LowerMin failed!");

    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint16_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint16_t, std::uint8_t>, std::uint16_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint16_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint16_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint16_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint16_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint16_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint16_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::LowerMin failed!");

    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int32_t, std::int8_t>, std::int32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int32_t, std::uint8_t>, std::int32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int32_t, std::int16_t>, std::int32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int32_t, std::uint16_t>, std::int32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int32_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int32_t, std::uint32_t>, std::int32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int32_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int32_t, std::uint64_t>, std::int32_t>::value,
        "alpaka::meta::LowerMin failed!");

    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint32_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint32_t, std::uint8_t>, std::uint32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint32_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint32_t, std::uint16_t>, std::uint32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint32_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint32_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint32_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint32_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::LowerMin failed!");

    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int64_t, std::int8_t>, std::int64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int64_t, std::uint8_t>, std::int64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int64_t, std::int16_t>, std::int64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int64_t, std::uint16_t>, std::int64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int64_t, std::int32_t>, std::int64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int64_t, std::uint32_t>, std::int64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int64_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int64_t, std::uint64_t>, std::int64_t>::value,
        "alpaka::meta::LowerMin failed!");

    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint64_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint64_t, std::uint8_t>, std::uint64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint64_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint64_t, std::uint16_t>, std::uint64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint64_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint64_t, std::uint32_t>, std::uint64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint64_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint64_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::LowerMin failed!");
}


BOOST_AUTO_TEST_SUITE_END()
