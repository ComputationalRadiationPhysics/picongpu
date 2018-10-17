/**
 * \file
 * Copyright 2018 Benjamin Worpitz
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

BOOST_AUTO_TEST_SUITE(core)

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(
    clipCastNoCastShouldNotChangeTheValue)
{
    BOOST_CHECK_EQUAL(
        std::numeric_limits<std::int8_t>::max(),
        alpaka::core::clipCast<std::int8_t>(std::numeric_limits<std::int8_t>::max()));
    BOOST_CHECK_EQUAL(
        std::numeric_limits<std::uint16_t>::min(),
        alpaka::core::clipCast<std::uint16_t>(std::numeric_limits<std::uint16_t>::min()));
    BOOST_CHECK_EQUAL(
        std::numeric_limits<std::int32_t>::min(),
        alpaka::core::clipCast<std::int32_t>(std::numeric_limits<std::int32_t>::min()));
    BOOST_CHECK_EQUAL(
        std::numeric_limits<std::uint64_t>::max(),
        alpaka::core::clipCast<std::uint64_t>(std::numeric_limits<std::uint64_t>::max()));
}

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(
    clipCastUpCastEqualSigndnessShouldNotChangeTheValue)
{
    BOOST_CHECK_EQUAL(
        static_cast<std::int16_t>(std::numeric_limits<std::int8_t>::max()),
        alpaka::core::clipCast<std::int16_t>(std::numeric_limits<std::int8_t>::max()));
    BOOST_CHECK_EQUAL(
        static_cast<std::uint32_t>(std::numeric_limits<std::uint16_t>::min()),
        alpaka::core::clipCast<std::uint32_t>(std::numeric_limits<std::uint16_t>::min()));
    BOOST_CHECK_EQUAL(
        static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::min()),
        alpaka::core::clipCast<std::int64_t>(std::numeric_limits<std::int32_t>::min()));
}

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(
    clipCastUpCastDifferentSigndnessShouldNotChangeTheValueForPositives)
{
    BOOST_CHECK_EQUAL(
        static_cast<std::uint16_t>(std::numeric_limits<std::int8_t>::max()),
        alpaka::core::clipCast<std::uint16_t>(std::numeric_limits<std::int8_t>::max()));
    BOOST_CHECK_EQUAL(
        static_cast<std::int32_t>(std::numeric_limits<std::uint16_t>::max()),
        alpaka::core::clipCast<std::int32_t>(std::numeric_limits<std::uint16_t>::max()));
    BOOST_CHECK_EQUAL(
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()),
        alpaka::core::clipCast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()));
}

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(
    clipCastUpCastDifferentSigndnessCanChangeTheValueForNegatives)
{
    BOOST_CHECK_EQUAL(
        std::numeric_limits<std::uint16_t>::min(),
        alpaka::core::clipCast<std::uint16_t>(std::numeric_limits<std::int8_t>::min()));
    BOOST_CHECK_EQUAL(
        static_cast<std::int32_t>(std::numeric_limits<std::uint16_t>::min()),
        alpaka::core::clipCast<std::int32_t>(std::numeric_limits<std::uint16_t>::min()));
    BOOST_CHECK_EQUAL(
        std::numeric_limits<uint64_t>::min(),
        alpaka::core::clipCast<std::uint64_t>(std::numeric_limits<std::int32_t>::min()));
}

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(
    clipCastDownCastEqualSigndnessCanChangeTheValue)
{
    BOOST_CHECK_EQUAL(
        std::numeric_limits<std::uint8_t>::max(),
        alpaka::core::clipCast<std::uint8_t>(std::numeric_limits<std::uint16_t>::max()));
    BOOST_CHECK_EQUAL(
        std::numeric_limits<std::int16_t>::min(),
        alpaka::core::clipCast<std::int16_t>(std::numeric_limits<std::int32_t>::min()));
    BOOST_CHECK_EQUAL(
        std::numeric_limits<std::uint16_t>::max(),
        alpaka::core::clipCast<std::uint16_t>(std::numeric_limits<std::uint64_t>::max()));
    BOOST_CHECK_EQUAL(
        std::numeric_limits<std::int8_t>::min(),
        alpaka::core::clipCast<std::int8_t>(std::numeric_limits<std::int64_t>::min()));
}

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(
    clipCastDownCastDifferentSigndnessCanChangeTheValue)
{
    BOOST_CHECK_EQUAL(
        std::numeric_limits<std::int8_t>::max(),
        alpaka::core::clipCast<std::int8_t>(std::numeric_limits<std::uint16_t>::max()));
    BOOST_CHECK_EQUAL(
        std::numeric_limits<std::uint16_t>::min(),
        alpaka::core::clipCast<std::uint16_t>(std::numeric_limits<std::int32_t>::min()));
    BOOST_CHECK_EQUAL(
        static_cast<std::int16_t>(std::numeric_limits<std::uint64_t>::min()),
        alpaka::core::clipCast<std::int16_t>(std::numeric_limits<std::uint64_t>::min()));
    BOOST_CHECK_EQUAL(
        std::numeric_limits<std::uint8_t>::max(),
        alpaka::core::clipCast<std::uint8_t>(std::numeric_limits<std::int64_t>::max()));
}

BOOST_AUTO_TEST_SUITE_END()
