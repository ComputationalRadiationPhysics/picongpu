/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/core/ClipCast.hpp>

#include <catch2/catch.hpp>

//-----------------------------------------------------------------------------
TEST_CASE(
    "clipCastNoCastShouldNotChangeTheValue", "[core]")
{
    CHECK(
        std::numeric_limits<std::int8_t>::max() ==
        alpaka::core::clipCast<std::int8_t>(std::numeric_limits<std::int8_t>::max()));
    CHECK(
        std::numeric_limits<std::uint16_t>::min() ==
        alpaka::core::clipCast<std::uint16_t>(std::numeric_limits<std::uint16_t>::min()));
    CHECK(
        std::numeric_limits<std::int32_t>::min() ==
        alpaka::core::clipCast<std::int32_t>(std::numeric_limits<std::int32_t>::min()));
    CHECK(
        std::numeric_limits<std::uint64_t>::max() ==
        alpaka::core::clipCast<std::uint64_t>(std::numeric_limits<std::uint64_t>::max()));
}

//-----------------------------------------------------------------------------
TEST_CASE(
    "clipCastUpCastEqualSigndnessShouldNotChangeTheValue", "[core]")
{
    CHECK(
        static_cast<std::int16_t>(std::numeric_limits<std::int8_t>::max()) ==
        alpaka::core::clipCast<std::int16_t>(std::numeric_limits<std::int8_t>::max()));
    CHECK(
        static_cast<std::uint32_t>(std::numeric_limits<std::uint16_t>::min()) ==
        alpaka::core::clipCast<std::uint32_t>(std::numeric_limits<std::uint16_t>::min()));
    CHECK(
        static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::min()) ==
        alpaka::core::clipCast<std::int64_t>(std::numeric_limits<std::int32_t>::min()));
}

//-----------------------------------------------------------------------------
TEST_CASE(
    "clipCastUpCastDifferentSigndnessShouldNotChangeTheValueForPositives", "[core]")
{
    CHECK(
        static_cast<std::uint16_t>(std::numeric_limits<std::int8_t>::max()) ==
        alpaka::core::clipCast<std::uint16_t>(std::numeric_limits<std::int8_t>::max()));
    CHECK(
        static_cast<std::int32_t>(std::numeric_limits<std::uint16_t>::max()) ==
        alpaka::core::clipCast<std::int32_t>(std::numeric_limits<std::uint16_t>::max()));
    CHECK(
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ==
        alpaka::core::clipCast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()));
}

//-----------------------------------------------------------------------------
TEST_CASE(
    "clipCastUpCastDifferentSigndnessCanChangeTheValueForNegatives", "[core]")
{
    CHECK(
        std::numeric_limits<std::uint16_t>::min() ==
        alpaka::core::clipCast<std::uint16_t>(std::numeric_limits<std::int8_t>::min()));
    CHECK(
        static_cast<std::int32_t>(std::numeric_limits<std::uint16_t>::min()) ==
        alpaka::core::clipCast<std::int32_t>(std::numeric_limits<std::uint16_t>::min()));
    CHECK(
        std::numeric_limits<uint64_t>::min() ==
        alpaka::core::clipCast<std::uint64_t>(std::numeric_limits<std::int32_t>::min()));
}

//-----------------------------------------------------------------------------
TEST_CASE(
    "clipCastDownCastEqualSigndnessCanChangeTheValue", "[core]")
{
    CHECK(
        std::numeric_limits<std::uint8_t>::max() ==
        alpaka::core::clipCast<std::uint8_t>(std::numeric_limits<std::uint16_t>::max()));
    CHECK(
        std::numeric_limits<std::int16_t>::min() ==
        alpaka::core::clipCast<std::int16_t>(std::numeric_limits<std::int32_t>::min()));
    CHECK(
        std::numeric_limits<std::uint16_t>::max() ==
        alpaka::core::clipCast<std::uint16_t>(std::numeric_limits<std::uint64_t>::max()));
    CHECK(
        std::numeric_limits<std::int8_t>::min() ==
        alpaka::core::clipCast<std::int8_t>(std::numeric_limits<std::int64_t>::min()));
}

//-----------------------------------------------------------------------------
TEST_CASE(
    "clipCastDownCastDifferentSigndnessCanChangeTheValue", "[core]")
{
    CHECK(
        std::numeric_limits<std::int8_t>::max() ==
        alpaka::core::clipCast<std::int8_t>(std::numeric_limits<std::uint16_t>::max()));
    CHECK(
        std::numeric_limits<std::uint16_t>::min() ==
        alpaka::core::clipCast<std::uint16_t>(std::numeric_limits<std::int32_t>::min()));
    CHECK(
        static_cast<std::int16_t>(std::numeric_limits<std::uint64_t>::min()) ==
        alpaka::core::clipCast<std::int16_t>(std::numeric_limits<std::uint64_t>::min()));
    CHECK(
        std::numeric_limits<std::uint8_t>::max() ==
        alpaka::core::clipCast<std::uint8_t>(std::numeric_limits<std::int64_t>::max()));
}
