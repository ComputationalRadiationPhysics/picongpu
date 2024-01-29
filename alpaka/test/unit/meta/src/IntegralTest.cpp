/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/meta/Integral.hpp>

#include <catch2/catch_test_macros.hpp>

#include <type_traits>

TEST_CASE("isIntegralSupersetTrue", "[meta]")
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

TEST_CASE("isIntegralSupersetNoIntegral", "[meta]")
{
    static_assert(
        !alpaka::meta::IsIntegralSuperset<float, std::uint8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint64_t, double>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
}

TEST_CASE("isIntegralSupersetFalse", "[meta]")
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

TEST_CASE("higherMax", "[meta]")
{
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int8_t, std::int8_t>, std::int8_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int8_t, std::uint8_t>, std::uint8_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int8_t, std::int16_t>, std::int16_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int8_t, std::uint16_t>, std::uint16_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int8_t, std::int32_t>, std::int32_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int8_t, std::uint32_t>, std::uint32_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int8_t, std::int64_t>, std::int64_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int8_t, std::uint64_t>, std::uint64_t>,
        "alpaka::meta::HigherMax failed!");

    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint8_t, std::int8_t>, std::uint8_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint8_t, std::uint8_t>, std::uint8_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint8_t, std::int16_t>, std::int16_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint8_t, std::uint16_t>, std::uint16_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint8_t, std::int32_t>, std::int32_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint8_t, std::uint32_t>, std::uint32_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint8_t, std::int64_t>, std::int64_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint8_t, std::uint64_t>, std::uint64_t>,
        "alpaka::meta::HigherMax failed!");

    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int16_t, std::int8_t>, std::int16_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int16_t, std::uint8_t>, std::int16_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int16_t, std::int16_t>, std::int16_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int16_t, std::uint16_t>, std::uint16_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int16_t, std::int32_t>, std::int32_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int16_t, std::uint32_t>, std::uint32_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int16_t, std::int64_t>, std::int64_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int16_t, std::uint64_t>, std::uint64_t>,
        "alpaka::meta::HigherMax failed!");

    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint16_t, std::int8_t>, std::uint16_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint16_t, std::uint8_t>, std::uint16_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint16_t, std::int16_t>, std::uint16_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint16_t, std::uint16_t>, std::uint16_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint16_t, std::int32_t>, std::int32_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint16_t, std::uint32_t>, std::uint32_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint16_t, std::int64_t>, std::int64_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint16_t, std::uint64_t>, std::uint64_t>,
        "alpaka::meta::HigherMax failed!");

    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int32_t, std::int8_t>, std::int32_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int32_t, std::uint8_t>, std::int32_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int32_t, std::int16_t>, std::int32_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int32_t, std::uint16_t>, std::int32_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int32_t, std::int32_t>, std::int32_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int32_t, std::uint32_t>, std::uint32_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int32_t, std::int64_t>, std::int64_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int32_t, std::uint64_t>, std::uint64_t>,
        "alpaka::meta::HigherMax failed!");

    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint32_t, std::int8_t>, std::uint32_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint32_t, std::uint8_t>, std::uint32_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint32_t, std::int16_t>, std::uint32_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint32_t, std::uint16_t>, std::uint32_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint32_t, std::int32_t>, std::uint32_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint32_t, std::uint32_t>, std::uint32_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint32_t, std::int64_t>, std::int64_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint32_t, std::uint64_t>, std::uint64_t>,
        "alpaka::meta::HigherMax failed!");

    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int64_t, std::int8_t>, std::int64_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int64_t, std::uint8_t>, std::int64_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int64_t, std::int16_t>, std::int64_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int64_t, std::uint16_t>, std::int64_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int64_t, std::int32_t>, std::int64_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int64_t, std::uint32_t>, std::int64_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int64_t, std::int64_t>, std::int64_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::int64_t, std::uint64_t>, std::uint64_t>,
        "alpaka::meta::HigherMax failed!");

    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint64_t, std::int8_t>, std::uint64_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint64_t, std::uint8_t>, std::uint64_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint64_t, std::int16_t>, std::uint64_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint64_t, std::uint16_t>, std::uint64_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint64_t, std::int32_t>, std::uint64_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint64_t, std::uint32_t>, std::uint64_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint64_t, std::int64_t>, std::uint64_t>,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMax<std::uint64_t, std::uint64_t>, std::uint64_t>,
        "alpaka::meta::HigherMax failed!");
}

TEST_CASE("lowerMax", "[meta]")
{
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int8_t, std::int8_t>, std::int8_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int8_t, std::uint8_t>, std::int8_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int8_t, std::int16_t>, std::int8_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int8_t, std::uint16_t>, std::int8_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int8_t, std::int32_t>, std::int8_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int8_t, std::uint32_t>, std::int8_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int8_t, std::int64_t>, std::int8_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int8_t, std::uint64_t>, std::int8_t>,
        "alpaka::meta::LowerMax failed!");

    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint8_t, std::int8_t>, std::int8_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint8_t, std::uint8_t>, std::uint8_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint8_t, std::int16_t>, std::uint8_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint8_t, std::uint16_t>, std::uint8_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint8_t, std::int32_t>, std::uint8_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint8_t, std::uint32_t>, std::uint8_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint8_t, std::int64_t>, std::uint8_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint8_t, std::uint64_t>, std::uint8_t>,
        "alpaka::meta::LowerMax failed!");

    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int16_t, std::int8_t>, std::int8_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int16_t, std::uint8_t>, std::uint8_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int16_t, std::int16_t>, std::int16_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int16_t, std::uint16_t>, std::int16_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int16_t, std::int32_t>, std::int16_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int16_t, std::uint32_t>, std::int16_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int16_t, std::int64_t>, std::int16_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int16_t, std::uint64_t>, std::int16_t>,
        "alpaka::meta::LowerMax failed!");

    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint16_t, std::int8_t>, std::int8_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint16_t, std::uint8_t>, std::uint8_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint16_t, std::int16_t>, std::int16_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint16_t, std::uint16_t>, std::uint16_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint16_t, std::int32_t>, std::uint16_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint16_t, std::uint32_t>, std::uint16_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint16_t, std::int64_t>, std::uint16_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint16_t, std::uint64_t>, std::uint16_t>,
        "alpaka::meta::LowerMax failed!");

    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int32_t, std::int8_t>, std::int8_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int32_t, std::uint8_t>, std::uint8_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int32_t, std::int16_t>, std::int16_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int32_t, std::uint16_t>, std::uint16_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int32_t, std::int32_t>, std::int32_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int32_t, std::uint32_t>, std::int32_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int32_t, std::int64_t>, std::int32_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int32_t, std::uint64_t>, std::int32_t>,
        "alpaka::meta::LowerMax failed!");

    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint32_t, std::int8_t>, std::int8_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint32_t, std::uint8_t>, std::uint8_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint32_t, std::int16_t>, std::int16_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint32_t, std::uint16_t>, std::uint16_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint32_t, std::int32_t>, std::int32_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint32_t, std::uint32_t>, std::uint32_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint32_t, std::int64_t>, std::uint32_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint32_t, std::uint64_t>, std::uint32_t>,
        "alpaka::meta::LowerMax failed!");

    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int64_t, std::int8_t>, std::int8_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int64_t, std::uint8_t>, std::uint8_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int64_t, std::int16_t>, std::int16_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int64_t, std::uint16_t>, std::uint16_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int64_t, std::int32_t>, std::int32_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int64_t, std::uint32_t>, std::uint32_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int64_t, std::int64_t>, std::int64_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::int64_t, std::uint64_t>, std::int64_t>,
        "alpaka::meta::LowerMax failed!");

    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint64_t, std::int8_t>, std::int8_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint64_t, std::uint8_t>, std::uint8_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint64_t, std::int16_t>, std::int16_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint64_t, std::uint16_t>, std::uint16_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint64_t, std::int32_t>, std::int32_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint64_t, std::uint32_t>, std::uint32_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint64_t, std::int64_t>, std::int64_t>,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMax<std::uint64_t, std::uint64_t>, std::uint64_t>,
        "alpaka::meta::LowerMax failed!");
}

TEST_CASE("higherMin", "[meta]")
{
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int8_t, std::int8_t>, std::int8_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int8_t, std::uint8_t>, std::uint8_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int8_t, std::int16_t>, std::int8_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int8_t, std::uint16_t>, std::uint16_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int8_t, std::int32_t>, std::int8_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int8_t, std::uint32_t>, std::uint32_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int8_t, std::int64_t>, std::int8_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int8_t, std::uint64_t>, std::uint64_t>,
        "alpaka::meta::HigherMin failed!");

    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint8_t, std::int8_t>, std::uint8_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint8_t, std::uint8_t>, std::uint8_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint8_t, std::int16_t>, std::uint8_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint8_t, std::uint16_t>, std::uint16_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint8_t, std::int32_t>, std::uint8_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint8_t, std::uint32_t>, std::uint32_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint8_t, std::int64_t>, std::uint8_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint8_t, std::uint64_t>, std::uint64_t>,
        "alpaka::meta::HigherMin failed!");

    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int16_t, std::int8_t>, std::int8_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int16_t, std::uint8_t>, std::uint8_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int16_t, std::int16_t>, std::int16_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int16_t, std::uint16_t>, std::uint16_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int16_t, std::int32_t>, std::int16_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int16_t, std::uint32_t>, std::uint32_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int16_t, std::int64_t>, std::int16_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int16_t, std::uint64_t>, std::uint64_t>,
        "alpaka::meta::HigherMin failed!");

    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint16_t, std::int8_t>, std::uint16_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint16_t, std::uint8_t>, std::uint16_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint16_t, std::int16_t>, std::uint16_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint16_t, std::uint16_t>, std::uint16_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint16_t, std::int32_t>, std::uint16_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint16_t, std::uint32_t>, std::uint32_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint16_t, std::int64_t>, std::uint16_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint16_t, std::uint64_t>, std::uint64_t>,
        "alpaka::meta::HigherMin failed!");

    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int32_t, std::int8_t>, std::int8_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int32_t, std::uint8_t>, std::uint8_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int32_t, std::int16_t>, std::int16_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int32_t, std::uint16_t>, std::uint16_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int32_t, std::int32_t>, std::int32_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int32_t, std::uint32_t>, std::uint32_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int32_t, std::int64_t>, std::int32_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int32_t, std::uint64_t>, std::uint64_t>,
        "alpaka::meta::HigherMin failed!");

    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint32_t, std::int8_t>, std::uint32_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint32_t, std::uint8_t>, std::uint32_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint32_t, std::int16_t>, std::uint32_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint32_t, std::uint16_t>, std::uint32_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint32_t, std::int32_t>, std::uint32_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint32_t, std::uint32_t>, std::uint32_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint32_t, std::int64_t>, std::uint32_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint32_t, std::uint64_t>, std::uint64_t>,
        "alpaka::meta::HigherMin failed!");

    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int64_t, std::int8_t>, std::int8_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int64_t, std::uint8_t>, std::uint8_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int64_t, std::int16_t>, std::int16_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int64_t, std::uint16_t>, std::uint16_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int64_t, std::int32_t>, std::int32_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int64_t, std::uint32_t>, std::uint32_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int64_t, std::int64_t>, std::int64_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::int64_t, std::uint64_t>, std::uint64_t>,
        "alpaka::meta::HigherMin failed!");

    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint64_t, std::int8_t>, std::uint64_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint64_t, std::uint8_t>, std::uint64_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint64_t, std::int16_t>, std::uint64_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint64_t, std::uint16_t>, std::uint64_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint64_t, std::int32_t>, std::uint64_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint64_t, std::uint32_t>, std::uint64_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint64_t, std::int64_t>, std::uint64_t>,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::HigherMin<std::uint64_t, std::uint64_t>, std::uint64_t>,
        "alpaka::meta::HigherMin failed!");
}

TEST_CASE("lowerMin", "[meta]")
{
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int8_t, std::int8_t>, std::int8_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int8_t, std::uint8_t>, std::int8_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int8_t, std::int16_t>, std::int16_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int8_t, std::uint16_t>, std::int8_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int8_t, std::int32_t>, std::int32_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int8_t, std::uint32_t>, std::int8_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int8_t, std::int64_t>, std::int64_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int8_t, std::uint64_t>, std::int8_t>,
        "alpaka::meta::LowerMin failed!");

    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint8_t, std::int8_t>, std::int8_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint8_t, std::uint8_t>, std::uint8_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint8_t, std::int16_t>, std::int16_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint8_t, std::uint16_t>, std::uint16_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint8_t, std::int32_t>, std::int32_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint8_t, std::uint32_t>, std::uint32_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint8_t, std::int64_t>, std::int64_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint8_t, std::uint64_t>, std::uint64_t>,
        "alpaka::meta::LowerMin failed!");

    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int16_t, std::int8_t>, std::int16_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int16_t, std::uint8_t>, std::int16_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int16_t, std::int16_t>, std::int16_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int16_t, std::uint16_t>, std::int16_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int16_t, std::int32_t>, std::int32_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int16_t, std::uint32_t>, std::int16_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int16_t, std::int64_t>, std::int64_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int16_t, std::uint64_t>, std::int16_t>,
        "alpaka::meta::LowerMin failed!");

    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint16_t, std::int8_t>, std::int8_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint16_t, std::uint8_t>, std::uint16_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint16_t, std::int16_t>, std::int16_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint16_t, std::uint16_t>, std::uint16_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint16_t, std::int32_t>, std::int32_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint16_t, std::uint32_t>, std::uint32_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint16_t, std::int64_t>, std::int64_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint16_t, std::uint64_t>, std::uint64_t>,
        "alpaka::meta::LowerMin failed!");

    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int32_t, std::int8_t>, std::int32_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int32_t, std::uint8_t>, std::int32_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int32_t, std::int16_t>, std::int32_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int32_t, std::uint16_t>, std::int32_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int32_t, std::int32_t>, std::int32_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int32_t, std::uint32_t>, std::int32_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int32_t, std::int64_t>, std::int64_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int32_t, std::uint64_t>, std::int32_t>,
        "alpaka::meta::LowerMin failed!");

    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint32_t, std::int8_t>, std::int8_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint32_t, std::uint8_t>, std::uint32_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint32_t, std::int16_t>, std::int16_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint32_t, std::uint16_t>, std::uint32_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint32_t, std::int32_t>, std::int32_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint32_t, std::uint32_t>, std::uint32_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint32_t, std::int64_t>, std::int64_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint32_t, std::uint64_t>, std::uint64_t>,
        "alpaka::meta::LowerMin failed!");

    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int64_t, std::int8_t>, std::int64_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int64_t, std::uint8_t>, std::int64_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int64_t, std::int16_t>, std::int64_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int64_t, std::uint16_t>, std::int64_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int64_t, std::int32_t>, std::int64_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int64_t, std::uint32_t>, std::int64_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int64_t, std::int64_t>, std::int64_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::int64_t, std::uint64_t>, std::int64_t>,
        "alpaka::meta::LowerMin failed!");

    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint64_t, std::int8_t>, std::int8_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint64_t, std::uint8_t>, std::uint64_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint64_t, std::int16_t>, std::int16_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint64_t, std::uint16_t>, std::uint64_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint64_t, std::int32_t>, std::int32_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint64_t, std::uint32_t>, std::uint64_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint64_t, std::int64_t>, std::int64_t>,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same_v<alpaka::meta::LowerMin<std::uint64_t, std::uint64_t>, std::uint64_t>,
        "alpaka::meta::LowerMin failed!");
}
